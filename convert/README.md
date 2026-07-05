# Conversion ACUITY : face_landmarker → NBG (NPU VIP9000 / Allwinner A733)

> **STATUT (13/06/2026) : chaîne validée de bout en bout.** Les 3 modèles
> passent import → quantize pcq int8 → export NBG avec zéro erreur.
> Artefacts archivés dans `../compiled/<model>_nbg_{fp16,int16}/`
> (`network_binary.nb` + projet C généré). Le dossier `work/` est un scratch
> de build (gitignoré) ; les livrables compilés sont déplacés dans
> `npu/compiled/`.
> Cible A733 : `--optimize VIP9000NANODI_PID0X1000003B`
> `--viv-sdk ~/Vivante_IDE/VivanteIDE5.11.0/cmdtools` (image
> `ubuntu-npu:v2.0.10.2`, ACUITY 6.30.22). Les wrappers `pegasus_*.sh` de la
> doc Radxa ne sont pas dans l'image : appeler `pegasus.py` directement
> (commandes ci-dessous mises à jour).
> **Validation numérique (13/06, vs TFLite de référence, même entrée)**, > grille complète des précisions (erreur moyenne absolue) :
>
> | sortie | float import | **fp16** | bf16 | int16 | pcq int8 |
> |---|---|---|---|---|---|
> | detector (logits) | 0,248 | 0,248 | 0,245 | 0,251 | 1,97 ❌ |
> | landmarks (px) | 0,089 | **0,093** | 0,217 | 0,122 | 1,78 ❌ |
> | blendshapes | 2e-5 | **2e-4** | 0,002 | 1e-4 | 0,0095 |
>
> **Choix : fp16 partout** (quantizer `float16`, non documenté chez Radxa
> mais présent dans ACUITY 6.30.22 ; c'est la précision native des modèles
> MediaPipe, sans perte, et le VIP9000 exécute le FP16 nativement).
> **int16 compilé en alternative vitesse** (l'int8/int16 a ~2× le débit fp16
> sur ce NPU, le benchmark on-device départagera). pcq int8 rejeté (dégrade
> detector et landmarks). NBG des deux variantes archivés dans
> `npu/compiled/<model>_nbg_{fp16,int16}/`.
>
> Versions des modèles : vérifié sur GCS, le bucket `mediapipe-models` ne
> contient que `float16/1/` = `latest` (2023-05-03, etag identique). Aucun
> bundle public plus récent n'existe ; le « blendshapes v3 » (issue MediaPipe
> #5329) n'a jamais été publié.
>
> ⚠️ Calibration des modes int8/int16 encore monotone (augmentations d'une
> seule photo), sans objet pour fp16 (pas de calibration nécessaire).

Plan de conversion des 3 modèles TFLite extraits de `face_landmarker.task`
(dans `npu/models/`) vers le format NBG exécutable par le NPU du Cubie A7A.

## Autopsie des modèles (12/06/2026)

| Modèle | Entrée | Sorties | Opérateurs | Verdict |
|---|---|---|---|---|
| `face_detector` (BlazeFace, 224 Ko) | `input` 1×128×128×3 f32 | `regressors` 1×896×16, `classificators` 1×896×1 | Conv2D, DWConv, ReLU, Add, Pad, MaxPool, Reshape, Concat | CNN pur ✅ |
| `face_landmarks_detector` (2,4 Mo) | `input_12` 1×256×256×3 f32 | `Identity` 1×1×1×1434 (478 pts ×3), `Identity_1` (présence), `Identity_2` | Conv2D, PReLU, DWConv, Add, MaxPool, Pad, Logistic, Reshape | CNN pur ✅ (pas d'op d'attention exotique) |
| `face_blendshapes` (933 Ko) | `serving_default_input_points:0` 1×146×2 f32 | `StatefulPartitionedCall:0` [52] | Conv 1×1, Mul/Add/Sub, Mean, SquaredDifference, Rsqrt, Transpose, StridedSlice, Logistic | MLP-Mixer, LayerNorm **déjà décomposé en primitives** ✅ |

Les ~370 `DEQUANTIZE` sont le stockage float16 des poids (motif standard des
modèles MediaPipe), à plier à l'import ou à neutraliser en re-sérialisant en
float32 si pegasus bronche.

## Étape 0, récupérer l'outillage (action manuelle, une fois)

L'image Docker ACUITY pour l'A733 est distribuée par Allwinner :

1. Ouvrir <https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq> dans
   un navigateur (interface Synology, téléchargement scripté impossible).
2. Télécharger `docker_images_v2.0.x.zip` (contient `ubuntu-npu:v2.0.10.1`).
3. Le déposer dans le dossier parent du projet, puis :

```bash
unzip docker_images_v2.0.x.zip
docker load -i ubuntu-npu_v2.0.10.1.tar     # nom exact selon l'archive
```

Note Apple Silicon : l'image est x86_64 → activer « Use Rosetta for x86_64
emulation » dans Docker Desktop (Settings → General). L'émulation suffit pour
la conversion (c'est du travail offline, la vitesse importe peu).

## Étape 1, workspace pegasus par modèle

Chaque modèle a son dossier (généré par `prepare_workspaces.py`) :

```
npu/convert/work/
├── face_detector/
│   ├── face_detector.tflite
│   ├── dataset.txt              # liste d'images de calibration
│   ├── inputs_outputs.txt       # noms des tenseurs d'E/S
│   └── calib/...                # crops 128×128
├── face_landmarks_detector/     # idem, crops 256×256
└── face_blendshapes/            # calibration = vecteurs de landmarks (npy)
```

Normalisation d'entrée (à reporter dans `*_inputmeta.yml` après l'import) :

| Modèle | mean | scale | note |
|---|---|---|---|
| face_detector | 127.5 127.5 127.5 | 1/127.5 ≈ 0.007843 | RGB → [-1, 1] |
| face_landmarks_detector | 0 0 0 | 1/255 ≈ 0.003922 | RGB → [0, 1] |
| face_blendshapes | 0 | 1.0 | landmarks bruts (pas une image) |

## Étape 2, séquence pegasus (dans le conteneur)

```bash
docker run --platform linux/amd64 --ipc=host -itd \
    -v $(pwd)/npu/convert/work:/workspace \
    --name acuity ubuntu-npu:v2.0.10.2 /bin/bash

# Dans le conteneur, pour chaque modèle (cwd = dossier du modèle) :
PEGASUS=~/acuity-toolkit-whl-6.30.22/bin/pegasus.py
cd /workspace/face_detector

python3 $PEGASUS import tflite --model face_detector.tflite \
    --output-model face_detector.json --output-data face_detector.data

python3 $PEGASUS generate inputmeta --model face_detector.json \
    --input-meta-output face_detector_inputmeta.yml
# éditer le yml : mean/scale (table ci-dessus) ; blendshapes : category
# undefined + preproc_type TENSOR

python3 $PEGASUS quantize --model face_detector.json \
    --model-data face_detector.data --with-input-meta face_detector_inputmeta.yml \
    --rebuild --model-quantize face_detector.quantize \
    --quantizer perchannel_symmetric_affine --qtype int8

python3 $PEGASUS export ovxlib --model face_detector.json \
    --model-data face_detector.data --model-quantize face_detector.quantize \
    --with-input-meta face_detector_inputmeta.yml --dtype quantized \
    --optimize VIP9000NANODI_PID0X1000003B \
    --viv-sdk ~/Vivante_IDE/VivanteIDE5.11.0/cmdtools \
    --pack-nbg-unify          # → ../face_detector_nbg_unify/network_binary.nb + C
```

Stratégie de quantization (cf. ../RESEARCH.md) : `pcq` ou `int16` d'emblée, l'exemple Radxa documente que l'uint8 naïf dégrade la précision. Si la
précision des landmarks chute : quantization hybride (page « Quantization
Precision Optimization » de Radxa), voire `bf16`.

## Étape 3, validation numérique

Comparer, sur les mêmes entrées : sorties TFLite/XNNPACK (référence CPU) vs
sorties pegasus_inference quantifiées. Critère : erreur moyenne sur les 478
landmarks < 1 px à 256×256, et erreur absolue < 0,02 sur les 52 blendshapes
(l'hystérésis de l'app absorbe ce bruit sans changer les décisions).

## Étape 4, intégration C (après validation)

`pegasus_export_ovx` génère un projet C OpenVX ; les exemples du Model Zoo
Radxa (C++/VIPLite, CMake) servent de gabarit pour le runner on-device.
Pré/post-traitement à réimplémenter : letterbox + normalisation, décodage des
ancres BlazeFace (896 ancres SSD), sous-ensemble des 146 landmarks d'entrée du
modèle blendshapes, matrice de transformation faciale (géométrie pure,
`geometry_pipeline_metadata_landmarks.binarypb`).

## TODO connus

- [ ] Extraire la liste des 146 indices de landmarks (sous-ensemble des 478)
      que MediaPipe donne au modèle blendshapes (source MediaPipe,
      `face_blendshapes_graph`).
- [ ] Données de calibration réelles : crops de visages variés (distances,
      éclairages) capturés par la caméra de l'installation.
- [ ] Mesurer le mode `float` (sans quantization), possiblement FP16 natif,
      vitesse inconnue (question ouverte de RESEARCH.md).
