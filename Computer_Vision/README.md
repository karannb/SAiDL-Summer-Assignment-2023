# <u>SAiDL Summer 2023 Induction Assignment Task-2 Computer Vision </u> <br></br>
## _Zero-Shot Segmentation using OpenAI's CLIP as an Encoder_ <br></br>

## _Task_ -
### Create and train a decoder model on top of CLIP's encoder to segment images based on random image-text prompts. Had to try models of different complexities, different loss functions and use 2-3 evaluation metrics. <br></br>

## _Bonus task_ -
### Prove/Disprove the hypothesis - _"Pixel intensities in a grayscale image vary smoothly within an object, as opposed intensities from pixels that cross object boundaries or belong to different objects.”_, using statistical tests (like Welch's T-test).<br></br>

## _Approach :-_
## 1. Data -
The first task I did was to find a way to navigate the dataset and generate input-output pairs of image-masks. I extracted the data out of the huge json file using pandas, dropped un-necessary columns and made a custom mask generator [`ground_images.py`](/Desktop/SAiDL/SAiDL%20Summer%202023/Computer_Vision/ground_images.py) and stored them in a separate folders. 
### Eg - 
#### Prompt is "Red Cushions"
#### input
![](display/4346.jpg)
#### output
![](display/4346__1558040.jpg)

## 2. Models - 
Now I decided a basic model architecture (model0) that I tried out with all 3 loss functions and then picked the loss function from the [CLIPSeg paper](https://arxiv.org/pdf/2112.10003.pdf) (BCELoss) to try out with other models (model1, model2). Description of the models -
### **model0** - 
layers with FiLM Conditioning<sup>[3](#notes)</sup> = "all" <br>
layers extracted from CLIP = total 5, (2nd, 4th, 6th, 8th, 10th) <br> <details>
<summary>Loss vs Epoch graph</summary>

![BCE](BCE.png)
</details><details>
<summary>Architecture</summary>

![add architecture here]()
</details><details>
<summary>Over the iterations</summary>

**Input<sup>[4](#notes)</sup>** (Prompt - "Giraffe")<br>
![](display/2315419.jpg)<br>
**Epoch 1** <br>
![](iterations_f/Focal1.jpg)<br>
**Epoch 2** <br>
![](iterations_f/Focal2.jpg)<br>
**Epoch 3** <br>
![](iterations_f/Focal3.jpg)<br>
**Epoch 4** <br>
![](iterations_f/Focal4.jpg)<br>
**Epoch 5** <br>
![](iterations_f/Focal5.jpg)<br>
**Epoch 6** <br>
![](iterations_f/Focal6.jpg)<br>
**Epoch 7** <br>
![](iterations_f/Focal7.jpg)<br>
**Epoch 8** <br>
![](iterations_f/Focal8.jpg)<br>
**Epoch 9** <br>
![](iterations_f/Focal9.jpg)<br>
**Epoch 10** <br>
![](iterations_f/Focal10.jpg)<br>
</details> <br>

### **model1** - (original CLIPSeg implementation)
layers with FiLM Conditioning<sup>[3](#notes)</sup> = final layer <br>
layers extracted from CLIP = total 3, (3rd, 6th, 9th) <br><details>
<summary>Loss vs Epoch graph</summary>

![ClipSeg](ClipSeg.png)
</details><details>
<summary>Architecture</summary>

![add architecture here]()
</details><details>
<summary>Over the iterations</summary>

**Input** (Prompt - "Goat")<br>
![](display/2317187.jpg)<br>
**Epoch 1** <br>
![](iterations_cs/ClipSege1.jpg)<br>
**Epoch 2** <br>
![](iterations_cs/ClipSege2.jpg)<br>
**Epoch 3** <br>
![](iterations_cs/ClipSege3.jpg)<br>
**Epoch 4** <br>
![](iterations_cs/ClipSege4.jpg)<br>
**Epoch 5** <br>
![](iterations_cs/ClipSege5.jpg)<br>
**Epoch 6** <br>
![](iterations_cs/ClipSege6.jpg)<br>
**Epoch 7** <br>
![](iterations_cs/ClipSege7.jpg)<br>
**Epoch 8** <br>
![](iterations_cs/ClipSege8.jpg)<br>
**Epoch 9** <br>
![](iterations_cs/ClipSege9.jpg)<br>
**Epoch 10** <br>
![](iterations_cs/ClipSege10.jpg)<br>
</details> <br>

### **model2** - 
layers with FiLM Conditioning<sup>[3](#notes)</sup> = "all" <br>
layers extracted from CLIP = total 4, (8th, 9th, 10th, 11th) <br><details>
<summary>Loss vs Epoch graph</summary>

![model2](model2.png)
</details><details>
<summary>Architecture</summary>

![add architecture here]()
</details><details>
<summary>Over the iterations</summary>

**Input** (Prompt - "Goat")<br>
![](display/2317187.jpg)<br>
**Epoch 1** <br>
![](iterations_2/model2e1.jpg)<br>
**Epoch 2** <br>
![](iterations_2/model2e2.jpg)<br>
**Epoch 3** <br>
![](iterations_2/model2e3.jpg)<br>
**Epoch 4** <br>
![](iterations_2/model2e4.jpg)<br>
**Epoch 5** <br>
![](iterations_2/model2e5.jpg)<br>
**Epoch 6** <br>
![](iterations_2/model2e6.jpg)<br>
**Epoch 7** <br>
![](iterations_2/model2e7.jpg)<br>
**Epoch 8** <br>
![](iterations_2/model2e8.jpg)<br>
**Epoch 9** <br>
![](iterations_2/model2e9.jpg)<br>
**Epoch 10** <br>
![](iterations_2/model2e10.jpg)<br>
</details><br>

**Quantitative Results** - (On the validation set, after 10 epochs)
| Model | Pixel-by-Pixel Accuracy(%) | Dice Score(%) | IOU Accuracy(%) | Trainable parameters | Projected Dimension|
| :- | :-: | :-: | :-: | :-: | :-: | 
| model0 | 86.15869 | **24.86891** | 18.49219 | **3871425** | **128** |
| model1 | **86.20316** | 23.61466 | 17.79634 | 1128545 | 64 | 
| model2 | 86.20224 | 24.75474 | **18.49252** | 3179969 | **128** |
<br>

### **Qualitative Analysis** - 
There is <u>no significant difference</u> between the model performances (~<1% for all metrics) despite my model having about <u>_~2 million_</u> more parameters, I believe this is because CLIP's feature space is very rich and it captures the Dataset properly, so a simple decoder is more than adequate. I tried to make my model as complex as possible (model0) but I believe the CLIPSeg model would require maybe a small amount of extra training to reach similar performance while being light-weight during deployment. I would like to point out that My model's (model2) learning graph is much more smooth than other 2 models, convergence is evident. All 3 models predict very similar masks with some jitter.

### **A few Masks** - 
<details><summary>Example1</summary>

_Input_ (Prompt - "Airplane") <br>
![xyz](Masks_models/input-train-2369077-Airplane.jpg) <br>
_Output_ (order - model0 -> model1 -> model2) <br>
![model0](Masks_models/output-train-2369077-model0-Airplane.jpg)<br>
![model1](Masks_models/output-train-2369077-model1-Airplane.jpg)<br>
![model2](Masks_models/output-train-2369077-model2-Airplane.jpg)<br>
</details>
<details><summary>Example2</summary>

_Input_ (Prompt - "Mountains in the background") <br>
![xyz](Masks_models/input-train-2369981-Mountains_in_the_background.jpg) <br>
_Output_ (order - model0 -> model1 -> model2) <br>
![model0](Masks_models/output-2369981-model0-Mountains_in_the_background.jpg)<br>
![model1](Masks_models/output-2369981-model1-Mountains_in_the_background.jpg)<br>
![model2](Masks_models/output-2369981-model2-Mountains_in_the_background.jpg)<br>
</details>
<details><summary>Example3</summary>

_Input_ (Prompt - "Surfer with a blue surfboard") <br>
![xyz](Masks_models/input-train-2371306-Surfer_with_a_blue_surfboard.jpg) <br>
_Output_ (order - model0 -> model1 -> model2) <br>
![model0](Masks_models/output-train-2371306-model0-Surfer_with_a_blue_surfboard.jpg)<br>
![model1](Masks_models/output-train-2371306-model1-Surfer_with_a_blue_surfboard.jpg)<br>
![model2](Masks_models/output-train-2371306-model2-Surfer_with_a_blue_surfboard.jpg)<br>
</details>
<details><summary>Example4</summary>

_Input_ (Prompt - "stack of gifts") <br>
![xyz](Masks_models/input-val-2339423-stack_of_gifts.jpg) <br>
_Output_ (order - model0 -> model1 -> model2) <br>
![model0](Masks_models/output-val-2339423-model0-stack_of_gifts.jpg)<br>
![model1](Masks_models/output-val-2339423-model1-stack_of_gifts.jpg)<br>
![model2](Masks_models/output-val-2339423-model2-stack_of_gifts.jpg)<br>
</details>
<details><summary>Example5</summary>

_Input_ (Prompt - "Orange bus") <br>
![xyz](Masks_models/input-val-2370002-Orange_bus.jpg) <br>
_Output_ (order - model0 -> model1 -> model2) <br>
![model0](Masks_models/output-val-2370002-model0-Orange_bus.jpg)<br>
![model1](Masks_models/output-val-2370002-model1-Orange_bus.jpg)<br>
![model2](Masks_models/output-val-2370002-model2-Orange_bus.jpg)<br>
</details>

</br>

## 3. Comparing results from different Loss functions -
### a) Graphs -
<details><summary> <b>BCEWithlogitsLoss</b> </summary>

![BCE](BCE.png) <br>
</details>

<details><summary> <b>DICE score as Loss</b> </summary>

![DICE](DICE.png) <br>
</details>

<details><summary> <b>Focal Loss</b> </summary>

![FOCAL](Focal.png) <br>
</details>

### b) <b>Quantitative Results</b> - (Using model0, on the validation set, after 10 epochs)
| Loss Function | Pixel-by-Pixel Accuracy(%) | Dice Score(%) | IOU Accuracy(%) |
| :- | :-: | :-: | :-: |
| BCEWithLogitsLoss | **86.15869** | 24.86891 | 18.49219 |
| Dice score as Loss| 81.48438 | **32.59544** | **23.69885** |
| Focal Loss | 82.56705 | 22.78004 | 16.82022 |
<br>

### c) <b>Qulaitative analysis</b> - 
1. From the graphs, clearly if Dice Loss is used we lose all information about convergence of the model, although it's masks are pretty good.
2. Dice Loss has the best masks and generalises very well to unknown prompts+images, because of it's inherent objective of maximisation of Dice Score whereas BCE focuses on individual pixels and Focal loss also has BCE (<u>extra focus on mis-classifed pixels</u>).
3. The CLIPSeg paper <u>still uses BCE</u>, this is most likely because with enough training, their <a href = "https://arxiv.org/pdf/2110.08322.pdf">performance converges</a>, + by 1, it is much more easy to control and check the progress of the model.
4. Combining my results and the paper referenced in 3, Dice Score + BCE would be the <u>best loss function</u> to use for semantic segmentation, combining explainability and generalisation.

[<b>Reference for loss functions' code</b>](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py)
<br>

### d) <b>Over the Iterations </b> -
1. <details>
   <summary>Focal Loss</summary>

    _Input (Prompt - "Giraffe")_<br>
    ![ip](iterations_f/input.jpg)<br>
    _Epoch 1_<br>
    ![e1](iterations_f/Focal1.jpg)<br>
    _Epoch 2_<br>
    ![e2](iterations_f/Focal2.jpg)<br>
    _Epoch 3_<br>
    ![e3](iterations_f/Focal3.jpg)<br>
    _Epoch 4_<br>
    ![e4](iterations_f/Focal4.jpg)<br>
    _Epoch 5_<br>
    ![e5](iterations_f/Focal5.jpg)<br>
    _Epoch 6_<br>
    ![e6](iterations_f/Focal6.jpg)<br>
    _Epoch 7_<br>
    ![e7](iterations_f/Focal7.jpg)<br>
    _Epoch 8_<br>
    ![e8](iterations_f/Focal8.jpg)<br>
    _Epoch 9_<br>
    ![e9](iterations_f/Focal9.jpg)<br>
    _Epoch 10_<br>
    ![e10](iterations_f/Focal10.jpg)<br>
    _Output_<br>
    ![op](iterations_f/output.jpg)<br>
    </details>
2. <details>
   <summary>Dice Loss<sup><a href = #notes>5</a></sup></summary>

   _Input (Prompt - "Truck")_<br>
    ![ip](iterations_d/input.jpg)<br>
    _Epoch 1_<br>
    ![e1](iterations_d/DICEe1.jpg)<br>
    _Epoch 2_<br>
    ![e6](iterations_d/DICEe6.jpg)<br>
    _Epoch 7_<br>
    ![e9](iterations_d/DICEe9.jpg)<br>
    _Epoch 10_<br>
    ![e10](iterations_d/DICEe10.jpg)<br>
    _Output_<br>
    ![op](iterations_d/output.jpg)<br>
    </details>

## 4. Visualising a few masks - 
<details><summary>Example 1</summary>

_Input (Prompt = "chocolate")_ <br>

![xyz](Comparing-Masks/input-train-4620-rug.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-train-4620-BCE-rug.jpg)<br>
![DICE](Comparing-Masks/output-train-4620-Dice-rug.jpg)<br>
![Focal](Comparing-Masks/output-train-4620-Focal-rug.jpg)<br>
</details>
<details><summary>Example 2</summary>

_Input (Prompt = "rug")_ <br>

![xyz](Comparing-Masks/input-train-4823-chocolate.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-train-4823-BCE-chocolate.jpg)<br>
![DICE](Comparing-Masks/output-train-4823-Dice-chocolate.jpg)<br>
![Focal](Comparing-Masks/output-train-4823-Focal-chocolate.jpg)<br>
</details>
<details><summary>Example 3</summary>

_Input (Prompt = "Skiing man")_ <br>

![xyz](Comparing-Masks/input-train-2320852-Skiing_man.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-val-2320852-BCE-Skiing_man.jpg)<br>
![DICE](Comparing-Masks/output-val-2320852-Dice-Skiing_man.jpg)<br>
![Focal](Comparing-Masks/output-val-2320852-Focal-Skiing_man.jpg)<br>
</details>
<details><summary>Example 4</summary>

_Input (Prompt = "catcher")_ <br>

![xyz](Comparing-Masks/input-val-2385854-catcher.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-val-4620-BCE-catcher.jpg)<br>
![DICE](Comparing-Masks/output-val-4620-Dice-catcher.jpg)<br>
![Focal](Comparing-Masks/output-val-4620-Focal-catcher.jpg)<br>
</details>
<details><summary>Example 5</summary>

_Input (Prompt = "sign post")_ <br>

![xyz](Comparing-Masks/input-train-2361304-sign_post.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-val-2361304-BCE-sign_post.jpg)<br>
![DICE](Comparing-Masks/output-val-2361304-Dice-sign_post.jpg)<br>
![Focal](Comparing-Masks/output-val-2361304-Focal-sign_post.jpg)<br>
</details>
<details><summary>Example 6</summary>

_Input (Prompt = "planes")_ <br>

![xyz](Comparing-Masks/input-val-2336273-planes.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-val-2336273-BCE-planes.jpg)<br>
![DICE](Comparing-Masks/output-val-2336273-Dice-planes.jpg)<br>
![Focal](Comparing-Masks/output-val-2336273-Focal-planes.jpg)<br>
</details>
<details><summary>Example 7</summary>

_Input (Prompt = "bread")_ <br>

![xyz](Comparing-Masks/input-val-2416046-bread.jpg)<br>
_Output (BCE -> DICE -> Focal)_ <br>

![BCE](Comparing-Masks/output-val-2416046-BCE-bread.jpg)<br>
![DICE](Comparing-Masks/output-val-2416046-Dice-bread.jpg)<br>
![Focal](Comparing-Masks/output-val-2416046-Focal-bread.jpg)<br>
</details>

</br>

## _Bonus Part_ - 
| Dataset | t-value | degrees of freedom (DOF) |
| :------ | :-----: | :----------------------: |
| Validation | 3.803102 | 12843.217203 |
| Train | 6.235643 | 59410.942513 |
| t-value at ∞ DOF and 99.9% confidence interval<sup>[11](https://www.ttable.org)</sup> | 3.291 | - |
### For the bonus part, I believe that our hypothesis about variations in pixel intensities is <u>correct</u> (_“Pixel intensities in a grayscale image vary smoothly within an object, as opposed intensities from pixels that cross object boundaries or belong to different objects.”_), to prove this claim it is sufficient to show that the two populations<sup>[1](#notes)</sup> involved were from different distributions and this was clear from the results I got; from the Welch's T-test, I was able to conclude that the two populations have <u>different means</u> even at 99.9% confidence interval, therefore they are statistically very different and from different distributions; the code<sup>[2](#notes)</sup> is quite self explanatory.<br></br>

## _What I was able to learn_ - 
### I was able to learn a **LOT** about image segmentation, transfer learning, muti-modal learning, loss functions and pyTorch implementations.<br></br>

## _References<sup>[6](#notes)</sup>_ - 
1. [CLIPSeg paper + code](https://github.com/timojl/clipseg)
2. [Open AI's CLIP, connecting images and text](https://www.youtube.com/watch?v=fQyHEXZB-nM)
3. [HuggingFace version of CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
4. [Robustness of Loss Functions](https://arxiv.org/pdf/2110.08322.pdf)
5. [FiLM](https://distill.pub/2018/feature-wise-transformations/)
6. [Code for loss functions](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py)
7. [Dice Score](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)
8. [IOU](https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef)
9. [Focal Loss](https://www.youtube.com/watch?v=Y8_OVwK4ECk)
10. [Welch's T-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)
11. [T-table](https://www.ttable.org)


## Notes -
1. the two Populations are _first, the **absolute pixel intensity differences between pixels belonging to the same object**, and the second, **absolute pixel intensity differences between pixels belonging to different objects**_.

2. I <u>did not</u> use my own decoder just to get better and accurate results for the hypothesis, my decoder has just been trained for 10 epochs, and would not predict very accurate masks.

3. About [FiLM](https://distill.pub/2018/feature-wise-transformations/) (Feature-wise Linear Modulation), it is nothing but just the projection of the text embedding onto the reduced dimension space (at-least in the current context).

4. The outputs are from when I used model0 with <u>Focal Loss</u> because my images for BCE Loss got corrupted.

5. "Over the iterations'" section of Dice Loss is <u>not at all indicative</u> of how the model is learning.

6. I may have provided links in the document but this is a consolidated list.