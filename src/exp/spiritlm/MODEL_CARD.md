# Meta Spirit LM Model Card

## Model Details

*Note: Use of this model is governed by the FAIR Noncommercial Research License.*

Spirit LM is a multimodal language model that freely mixes text and speech. The model can be prompted with either text or speech and is capable of generating outputs in either modality, while preserving the expressivity of the input prompt. The model is also able to learn new tasks across modalities such as automatic speech recognition, text-to-speech, and speech classification in a few-shot manner.

## Model Developers
Meta

## Variations
Spirit LM comes in two versions: Spirit LM Base that uses speech phonetic tokens and Spirit LM Expressive that models expressivity using pitch and style tokens in addition to the phonetic tokens.

## Input
Models input text or speech or a mixed sequence of the two.

## Output
Models generate text or speech or a mixed sequence of the two.

## Model Architecture
### Speech Tokenizer
Spirit LM uses 3 types of speech tokenizers: Phonetic Tokenizer (HuBERT), Pitch Tokenizer (VQ-VAE) and Style Tokenizer (Speechprop or Wav2vec2). We use Hifi-GAN to convert the speech tokens back to audio.

It is worth noting that in the associated paper, for Spirit LM Expressive, we used Speechprop to extract style tokens, while we use a Wav2vec2 model to extract style tokens in this release.

|                        | Model                    | Parameters | Input               | Output             |
|------------------------|--------------------------|------------|---------------------|--------------------|
| Phonetic Tokenizer     | HuBERT+LinearQuantizer   | 96M        | Waveform            | Phonetic Tokens    |
| Pitch Tokenizer        | VQ-VAE                   | 0.2M       | Extracted F0        | Pitch Tokens       |
| Style Tokenizer        | Wav2vec2+LinearProjection| 95M        | Waveform            | Style Tokens       |
| Base Speech Decoder    | Hifi-GAN                 | 14M        | Phonetic Tokens     | Waveform           |
| Expressive Speech Decoder | Hifi-GAN              | 15M        | Phonetic, Pitch, Style Tokens | Waveform

### Language Model
Spirit LM is initialized from the Llama-2 7B model.

|                      | Architecture   | Parameters | Input/Output Tokens                                      | Vocab Size |
|----------------------|----------------|------------|----------------------------------------------------------|------------|
| Spirit LM Base       | Llama-2 7B     | 7B         | Text Tokens, Phonetic Tokens                             | 32512      |
| Spirit LM Expressive | Llama-2 7B     | 7B         | Text Tokens, Phonetic Tokens, Pitch Tokens, Style Tokens | 32768      |

### Release Date
The models were trained between October and December 2023. The research paper was released on February 8th 2024. We released the model on October 18th 2024.

### Status
This is a static model trained on an offline dataset.

### License
We release the model under the FAIR Noncommercial Research License found in the [LICENSE](LICENSE) file in the root directory of this repo.

### Research Paper
More information can be found in the paper ["SpiRit-LM: Interleaved Spoken and Written Language Model"](https://arxiv.org/pdf/2402.05755.pdf).

## Hardware and Software
### Training Factors
We used custom training libraries. The training of the released models has been performed on Meta’s Research Clusters.

The training of each model (Spirit LM Base and Spirit LM Expressive) takes 21K GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W), not including the training of Llama-2.

## Training Data
We trained the models on a combination of text-only datasets, speech-only datasets and aligned speech-text datasets. All the speech datasets are publicly available. Here are the statistics of the datasets we used:

|              | Hours | Speech Tokens | Text Tokens |
|--------------|-------|---------------|-------------|
| Speech-only  | 458K  | 28.2B         | -           |
| Speech+Text  | 111K  | 7.0B          | 1.4B        |
| Text-only    | -     | -             | 307B        |

## Evaluation Results
See evaluations for our models and detailed ablations in Section 4 and 5, and safety evaluations in Section 6 of the [research paper](https://arxiv.org/pdf/2402.05755.pdf).

## Intended Use
### Intended Use Cases
Spirit LM is intended for noncommercial research use in English.

### Out-of-Scope Uses
Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English. Use in any other way that is prohibited by the FAIR Noncommercial Research License and Acceptable Use Policy.

## Ethical Considerations and Limitations
This model is built on Llama 2 which carries risks with use.  Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios.  For these reasons, as with all LLMs, Llama 2’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts.  The model’s speech capabilities are designed to analyze speaker agnostic qualities of any input speech and output speech in one of four pre-set voices. The model is meant for use for noncommercial research purposes only and should not be deployed in any consumer-facing applications.