# [Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation](https://www.arxiv.org/abs/2508.16188)
**Weiting Tan, Jiachen Lian, Hirofumi Inaguma, Paden Tomasello, Philipp Koehn, Xutai Ma**

<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>
<a href="https://www.arxiv.org/abs/2508.16188" alt="paper"><img src="https://img.shields.io/badge/AVLM-Paper-D9AB42" /></a>
<a href="https://www.clsp.jhu.edu/" alt="jhu"><img src="https://img.shields.io/badge/Johns_Hopkins_University-BEC23F" /></a>
<a href="https://twitter.com/weiting_nlp">
  <img src="https://img.shields.io/twitter/follow/weiting_nlp?style=social&logo=twitter"
      alt="follow on Twitter"></a>
</p>

---

**AVLM** is a research project that targets modality fusion, integrating visual and speech representation into pre-trained SpeechLM for expressive generation.


## 🔧 Project Structure

```bash
AVLM/
├── scripts/  # main scripts
│   ├── avlm/ # folder for AVLM pretraining (with different fusion strategies)
│   ├── avlm_avsr/ # folder for fine-tuning AVLM to perform AVSR task
│   ├── avlm_emo/ # folder for fine-tuning AVLM for expressive speech generation
│   ├── global.sh # config script for paths
├── src/                     # Core source code
│   ├── data_utils/          # Data loading and preprocessing
│   ├── models/              # Customized SpiritLM model 
│   ├── exp/                 # spiritlm source code
│   ├── preprocess/          # video data preprocessing
│   ├── task/          # lighting trainer files
│   ├──├── avlm_iemocap_tune.py # fine-tune AVLM for expressive dialogue generation
│   ├──├── train_avlm.py # pre-train AVLM with differenet fusion strategies or fine-tune AVLM for AVSR task
```

---



### If you find our work useful, please cite:
```
@misc{tan2025seeingbelievingemotionawareaudiovisual,
      title={Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation}, 
      author={Weiting Tan and Jiachen Lian and Hirofumi Inaguma and Paden Tomasello and Philipp Koehn and Xutai Ma},
      year={2025},
      eprint={2508.16188},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.16188}, 
}
```
