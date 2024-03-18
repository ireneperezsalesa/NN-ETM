NN-ETM
==============

This is the companion code for the paper "NN-ETM: Enabling Safe Neural Network-Based Event-Triggering Mechanisms for Consensus Problems", by [Irene Perez-Salesa](https://ireneperezsalesa.github.io/), [Rodrigo Aldana-Lopez](https://rodrigoaldana.github.io/) and [Carlos Sagues](https://webdiis.unizar.es/~csagues/). The preprint can be found [here]().

In this work, we develop NN-ETM, a neural network-based event-triggering mechanism for multi-agent consensus problems. We aim to provide a general solution for the communication policy of the agents by means of the NN-ETM, in which the neural network is used to optimize the behavior of the setup, while ensuring stability guarantees for the consensus protocol.
By using different neural network architectures within the structure of the NN-ETM and tuning the parameters in the cost function (which represents the trade-off between the communication load and the performance of the consensus protocol), different communication policies can be learned.

Here, we provide our code to train and test the NN-ETM to facilitate the use of our proposal.


User requirements
-----------------

Our code has been tested in an Anaconda3 environment with Python 3.11 
and the Pytorch version 2.1.2.


Files
-----
- **main.py**: Test a trained NN-ETM model and plot results.
- **training.py**: Training loop for NN-ETM. 
- **pretrain.py**: Optional pretraining process. Learns a fixed-threshold policy
for NN-ETM. The learned weights are used as initialization for training.py.
- **model/model.py**: Contains the NN architecture used by training.py.
- **algorithms/consensus_alg.py**: Contains a linear dynamic consensus protocol
using NN-ETM, used by training.py and main.py.
- **utils/generate_data.py**: Generates reference signals for the consensus protocol.
- **histograms.py** and **plot_histograms.py**: Replicate the histogram experiment in the paper.


Training and testing NN-ETM
---------------------------

- To train the NN-ETM, run **training.py** (from command line: ``python training.py``). The neural network aims to optimize a cost function
``
cost = consensus_error + L * communication_rate
``.
To adjust the trade-off between consensus error and communication rate 
of the event-triggered consensus setup, tune the parameter ``L`` in the cost function. 
Increasing ``L`` results in less communication but higher error values.
By default, the neural network is initialized to a pretrained model (pretrain/m_500.pth) which
has learned a fixed-threshold event triggering policy. The trained models are stored in the 
"checkpoints" directory.

- To test a trained model, edit the "path_to_model" variable in **main.py** to select the
desired trained neural network and run it (from command line: ``python main.py``). The plotted results are stored in the "figs" directory by default.

- Different parameters can be edited for both files, such as the training configuration, 
the design constants in the NN-ETM and the gain in the consensus protocol. To edit the neural network model,
the consensus protocol or the reference signals, please refer to the file descriptions above.


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
