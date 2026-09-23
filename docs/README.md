# FlashLoop project page

This directory is a dependency-free static GitHub Pages site, following the academic project-page structure of [AlphaQ](https://superone77.github.io/AlphaQ/).

- `index.html`: page content and paper result table.
- `static/css/style.css`: responsive layout and visual styling.
- `static/js/main.js`: BibTeX copy button.
- `static/images/`: PNG renders of the PDF figures actually referenced by `iclr2027_conference.tex` in the user-provided `FlashLoop__Fast_and_Memory_Efficient_Looped_Transformers_via_Lazy_Updates (1).zip`; `demo-poster.webp` is a frame from the demo video.
- `static/media/flashloop-mmlu-demo-2x.mp4`: the requested edited demo.

The Paper button is disabled until a public paper link is available. The submission PDF is not part of the site.

Figure mappings from the source archive:

| Site file | ZIP member |
| --- | --- |
| `pareto_frontier_horizontal.png` | `pareto_frontier_horizontal.pdf` |
| `flashloop_motivation.png` | `flashloop_motivation.pdf` |
| `cross_model_recurrent_generalization.png` | `cross_model_recurrent_generalization.pdf` |
| `component_efficiency_ablation_8k.png` | `component_efficiency_ablation_8k.pdf` |
| `hyperparameter_efficiency_accuracy_8k.png` | `hyperparameter_efficiency_accuracy_8k.pdf` |
| `context_measured_efficiency_triptych.png` | `context_measured_efficiency_triptych.pdf` |
| `scaling_with_loops.png` | `scaling_with_loops.pdf` |

To preview locally from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. GitHub Pages can deploy this directory with the included workflow or serve `docs/` from the `main` branch.
