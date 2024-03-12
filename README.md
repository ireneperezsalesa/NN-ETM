NN-ETM
==============

This is the companion code for the paper "NN-ETM: Enabling Safe Neural Network-Based Event-Triggering Mechanisms for Consensus Problems", by [Irene Perez-Salesa](https://ireneperezsalesa.github.io/), [Rodrigo Aldana-Lopez](https://rodrigoaldana.github.io/) and [Carlos Sagues](https://webdiis.unizar.es/~csagues/). The preprint can be found [here]().

In this work, we develop NN-ETM, a neural network-based event-triggering mechanism for multi-agent consensus problems. We aim to provide a general solution for the communication policy of the agents by means of the NN-ETM, in which the neural network is used to optimize the behavior of the setup, while ensuring stability guarantees for the consensus protocol.
By using different neural network architectures within the structure of the NN-ETM and tuning the parameters in the cost function (which represents the trade-off between the communication load and the performance of the consensus protocol), different communication policies can be learned.

Here, we provide our code to train and test the NN-ETM to facilitate the use of our proposal.


User requirements
-----------------

Our code has been tested in an Anaconda3 environment with Python 3.11 and the Pytorch version 2.1.2.


Citation
--------

If you find our proposal/code useful for your research, please cite our works as follows:

```bibtex
@article{nnetm2024,
  title = {{NN-ETM: Enabling Safe Neural Network-Based Event-Triggering Mechanisms for Consensus Problems}},
  author = {Irene Perez-Salesa and Rodrigo Aldana-Lopez and Carlos Sagues},
  journal = {ArXiv preprint ???},
  year = {2024}
}
```


Acknowledgments
---------------

This work was supported via projects PID2021-124137OB-I00 and TED2021-130224B-I00 funded by MCIN/AEI/10.13039/501100011033, by ERDF A way of making Europe and by the European Union NextGenerationEU/PRTR, by the Gobierno de Aragón under Project DGA T45-23R, by the Universidad de Zaragoza and Banco Santander, by the Consejo Nacional de Ciencia y Tecnología (CONACYT-México) grant 739841, and by Spanish grant FPU20/03134.
