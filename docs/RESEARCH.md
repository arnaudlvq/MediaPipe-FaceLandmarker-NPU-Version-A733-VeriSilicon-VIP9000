# Porter MediaPipe FaceLandmarker sur le NPU VIP9000 de l'Allwinner A733 — état de l'art

*Recherche approfondie du 12 juin 2026 — 5 axes, 19 sources primaires, 25 affirmations
vérifiées par triple vote adversarial (24 confirmées, 1 réfutée).*

## Verdict global

**Personne ne l'a fait publiquement.** Aucun portage de BlazeFace, Face Mesh /
Attention Mesh ou Blendshapes GHUM sur un NPU VeriSilicon VIP9000 n'existe — ni chez
Radxa (le [Model Zoo officiel A7A](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/model-zoo)
compte ~20 modèles : YOLO v3→v26, RetinaFace, MobileNet, ResNet50… zéro modèle
MediaPipe), ni dans les communautés Khadas (A311D), NXP (i.MX 8M Plus) ou Mesa/Teflon.
Le prior art le plus proche est RetinaFace (détection de visage seule). La page
« MediaPipe » des docs Radxa A7A est un simple `pip install mediapipe` CPU.

→ Ce portage serait une **première publique**, d'où l'intérêt d'un repo de qualité.

## Les quatre voies

| Voie | Verdict | Détail |
|---|---|---|
| **(a) ACUITY → NBG + VIPLite (C)** | ✅ **Mature, documentée, recommandée** | Seule chaîne documentée de bout en bout par Radxa pour le Cubie A7A. Accepte le TFLite en entrée (pas besoin de passer par ONNX). Scripts `pegasus_import/quantize/inference/export_ovx`. Génère un **projet OpenVX en C** — aligné avec l'objectif de réécriture C. Pas de chemin Python on-device documenté ; les exemples du Model Zoo sont en C++ contre VIPLite 2.0.3.2-AW. |
| **(b) TIM-VX + tflite-vx-delegate** | ⚠️ **Réelle mais non prouvée sur A733** | Delegate TFLite officiel VeriSilicon, utilisable depuis Python (`load_delegate`), démontré sur Khadas VIM3/A311D, production chez NXP. Supporte VIPLite via le SDK « no-kernel » (≥6.4.22). MAIS : l'A733 n'est pas une carte de référence, le build exige les libs userspace du BSP Allwinner (versions strictement appariées), et des issues ouvertes signalent sorties erronées/segfaults sur certains modèles. |
| **(c) Mesa Teflon / etnaviv** | ❌ **Morte pour l'A733** | Pile 100 % open-source mais limitée à VIPNano-QI (A311D), VIPNano-SI+ (i.MX 8M Plus) et RK3588 ; CNN UINT8 uniquement ; le VIP9000 de l'A733 n'est pas couvert, et Tomeu Vizoso a pivoté vers Rockchip en 2025. À surveiller à long terme. |
| **(d) ONNX Runtime / autres** | ❌ **Inexistant** | Pas d'execution provider VIPLite — c'est l'objet de l'[issue ouverte #28244](https://github.com/microsoft/onnxruntime/issues/28244) qui demande exactement le support A733/T527. |

## Le risque technique n° 1 : la couverture d'opérateurs

Les ops CNN de **BlazeFace** et du tronc Face Mesh passent partout (Conv2d,
DepthwiseConv2d, Prelu, Reshape, Pad, FullyConnected : tous « yes » dans
[op_status.md](https://github.com/VeriSilicon/tflite-vx-delegate/blob/main/op_status.md)).
En revanche **GELU et LayerNorm sont absents** de la matrice d'opérateurs du delegate
(vérifié dans le code : zéro occurrence dans op_map.cc) — or ce sont les briques des
mécanismes d'attention d'**Attention Mesh** et du **Blendshapes GHUM** (type MLP-Mixer).
La faisabilité de conversion de ces deux modèles n'est tranchée nulle part : **seul un
essai réel de `pegasus_import` le dira**.

Plans de repli si ça bloque : (1) ne porter que BlazeFace + landmarks et garder le
modèle blendshapes sur CPU (il prend les *landmarks* en entrée, pas l'image — il est
minuscule) ; (2) substituer Face Mesh V1 (sans attention) au V2.

## Quantization : la stratégie

Modes ACUITY documentés sur A7A : `uint8`, `pcq` (int8 per-channel), `int16`, `bf16`,
`float` (aucune quantization), + quantization hybride. **FP16 n'est pas une cible
explicite** (les modèles MediaPipe sont livrés en float16) — les options 16 bits sont
BF16/INT16, ou le mode `float` dont l'exécution interne n'est pas spécifiée.

Piège documenté par Radxa lui-même : dans leur exemple MobileNetV2, **l'uint8 simple
dégrade la précision** (mauvaise classe prédite), tandis que `pcq` et `int16` restent
conformes au float. Pour des modèles de *régression de landmarks* (plus sensibles
qu'un classifieur), partir directement sur **pcq ou int16**, avec la
[page d'optimisation de précision](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-quant-acc-improve)
(quantization mixte) en recours.

## Performances : aucune donnée publiée — on défrichera

Aucun chiffre de latence n'existe pour des modèles type BlazeFace/Face Mesh sur le
VIP9000 3 TOPS de l'A733. Seules références de la famille : A311D 5 TOPS, MobileNetV1
≈ 5,5-6,6 ms. Question ouverte assumée : pour des modèles aussi petits (1-5 MFLOPs),
le gain NPU dépasse-t-il le coût de transfert mémoire face aux A76 + XNNPACK ? **C'est
exactement ce que notre banc (`npu/bench/`) mesurera** — latence p50/p95/p99, fps,
températures, en pic et en endurance 2 h, CPU vs NPU sur la même grille. Outils
on-device complémentaires : `vpm_run` et `NBinfo` (documentés par Radxa, aucun
résultat publié trouvé).

## Réserves d'honnêteté

- La conversion d'Attention Mesh et du Blendshapes GHUM est **non prouvée** — risque
  principal, à lever en premier (essai de conversion avant toute écriture de code C).
- Le mode `float` d'ACUITY tourne peut-être en FP16 natif sur le NPU — vitesse inconnue.
- Les chiffres A311D ne se transposent pas directement (génération NPU différente).
- « Documenté » ≠ « fiable » : issues ouvertes sur le delegate (sorties erronées INT8,
  segfaults sur certains modèles).
- Docs Radxa sujettes au link rot (une URL a changé pendant la recherche même).

## Sources principales

[Radxa NPU dev (A7A)](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev) ·
[ACUITY usage](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-usage) ·
[Model Zoo A7A](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/model-zoo) ·
[tflite-vx-delegate](https://github.com/VeriSilicon/tflite-vx-delegate) ·
[TIM-VX](https://github.com/VeriSilicon/TIM-VX) ·
[Khadas VIM3 vx-tflite](https://docs.khadas.com/products/sbc/vim3/npu/vx-tflite) ·
[Mesa Teflon](https://docs.mesa3d.org/teflon.html) ·
[Blog Tomeu Vizoso](https://blog.tomeuvizoso.net/) ·
[onnxruntime #28244](https://github.com/microsoft/onnxruntime/issues/28244) ·
[Frigate #23418](https://github.com/blakeblackshear/frigate/discussions/23418) ·
[acuitylite (VeriSilicon)](https://verisilicon.github.io/acuitylite/README.html)
