# SAM in LTR
We addressed challenges in visual classification tasks due to long-tailed distribution data by proposing a novel self-attention-based network for long-tail recognition. Inspired by decoupled structures and the efficacy of attention mechanisms in visual tasks, our approach, **Self-Attention Mechanism(SAM)**, integrates tail-class feature information into deep networks. Surprisingly, incorporating a self-attention layer significantly enhances the network's robustness in recognizing long-tail distributions. Our simple yet effective feature extraction method from tail classes, coupled with extensive experimental evaluations, highlights the unique advantages of our model over traditional attention mechanisms in handling long-tailed recognition. 

# Tiny-ImageNet-LT
![image](https://github.com/FZKChange/SAM_in_LTR/assets/78149508/aca9ba64-ccd5-416b-8bd9-ff82304472dc)

The black curves in the figures illustrate the long-tail data distribution. The orange bars show the performance of the standard ResNet-50 model, while the blue bars represent the performance of our approach. **As depicted, the blue bars generally surpass the orange bars, demonstrating that our method not only improves tail class accuracy but also maintains head class accuracy effectively.**

# CIFAR-10-LT and CIFAR-100-LT
![image](https://github.com/FZKChange/SAM_in_LTR/assets/78149508/86a84026-5b5d-47db-bceb-4670df6aa80e)

Benchmark results from CIFAR-10-LT and CIFAR-100-LT datasets, as shown in the table, demonstrate that our method consistently outperforms the baseline unmodified deep network and most comparative methods. **For instance, on CIFAR100-LT, our approach reached 53.34%, surpassing the Pure model at 34.25% and other techniques like RISDA at 50.16% and ResLT at 48.21%.**
