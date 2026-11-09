# Citation

If you use Tachymetry in your research, please cite it:

```bibtex
@software{turkcan2026tachymetry,
  author  = {Turkcan, Mehmet Kerem},
  title   = {{Tachymetry}: Clean Metric Trajectories, Speeds and Counts from Traffic-Camera Video},
  year    = {2026},
  url     = {https://github.com/mkturkcan/tachymetry}
}
```

Results should also credit the detector that produced them: [RF-DETR](https://github.com/roboflow/rf-detr) for the default `rfdetr-seg-*` and `rfdetr-*` detectors, [SAM 3](https://github.com/facebookresearch/sam3) for `sam3`, or [UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2) for `uod2-*`; their repositories and model cards give the references.

## Sample footage

The demo, the examples and the figures of this documentation use public clips that are downloaded on demand from their hosts and not redistributed with Tachymetry. Figures show frames with Tachymetry's overlays added.

- [svbrd-llm-roadside-video-av](https://huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av) by Ryan-xiangyu-ut (Hugging Face), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): the five `intersection-*` clips.
- [Roundabout 14 61](https://commons.wikimedia.org/wiki/File:Roundabout_14_61.webm) by Wikideas1 (Wikimedia Commons), [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): `roundabout-aerial`.
- [Small-Crowd-and-Traffic-Video-Footages](https://huggingface.co/datasets/parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages) by parsafaramarzi (Hugging Face), MIT: `motorway-overpass`.
- [DRINF: Dhaka Road Intersection Nadir Footage](https://huggingface.co/datasets/hamimmahmud0/DRINF) by Hamim Mahmud (Hugging Face, doi:10.57967/hf/10301), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): `dhaka-roundabout`, a 30 s excerpt.

`tachymetry.data.attribution(name)` returns the credit line of each clip ([Sample clips](samples.md)). The footage credits of the trailer are listed in [docs/media/CREDITS.md](https://github.com/mkturkcan/tachymetry/blob/main/docs/media/CREDITS.md).

## Author

[Mehmet Kerem Turkcan](https://mehmetkeremturkcan.com/), Associate Research Scientist, Department of Civil Engineering, Columbia University.
