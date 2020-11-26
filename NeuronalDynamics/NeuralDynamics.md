Neuronal Dynamics by gerstner, et al.
=====================================

Notes.

Chapter 1
---------

### Introduction: Neurons and Mathematics

![Neural Network](https://neuronaldynamics.epfl.ch/online/x1.png)
Neurons are divided into 3 parts:
1. Dendrites
    * Neuron input.
2. Soma
    * The neuron's "CPU"
    * When total input *exceeds a certain threshold*, an output signal (Action Potential) is sent. This is a non-linear calculation.
3. Axon
    * Transmits the signal sent by the Soma.

A synapse is the connection of an axon and a dendrite.

Sending Neuron is refered as **Presynaptic neuron** and the recieving one **Postsynaptic neuron**

![Neuron structure](https://neuronaldynamics.epfl.ch/online/x2.png)

Action Potential == Pulses == Signals == Spikes

_Spike Train_: A sequence of action potential emited by a single neuron.

In neurons, Action Potential (spikes) have *no unique shapes, they are all the same*. This means that the shape of a spike does not communicate information, rather it is the number and timing of the spikes that tell us information.

Action Potentials cannot be triggered either on top or immediately after one. This means that there is always a period in the neuron where the cell so-to-say is recharging and no matter the strength of inputs, it is impossible to have another spike be triggered. this is an **Absolute Refactiry Period.** Right after this period, another period of semi-refactoriness (semi-recharge) occurs where it is _quite hard but not impossible for a new action potential to happen_.

_Presynaptic Potential_: The spike in the presynaptic neuron.
_Postsynaptic Potential_: The signal in the Post synaptic neuron. (triggered by the signal sent by the presynaptic neuron).

_Receptive Field_: The zone/area of stimuli where a neuron activates. (pixel 12, 4 for neuron 52!)

_Membrane Potential_: Within a neuron, the potential difference (the electrical level) is the function for the activity in a neuron u(t). A neuron is usually in u_rest, but when a signal comes in, it triggers a spike. If a change in u(t) is positive, we call the synapse to be **excitatory**, if it is negative, the synapse is **inhibitory**.

An input at an excitatory synapse that reduces polatization (since cell mebrane is naturally negativelly charged ~-65mV) it is called _depolarizing_ (V > -65mV), if it increases the negativity, it is called _hyperpolarizing_.

**Postsynaptic Potential (PSP) formula**
for a time course (change in membrane potential) u, in the neuron i, we denote this the change in membrane potetial in neuron *i* as ![u_i(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29) We let *j* be the Presynaptic neuron.

Let t=0 the moment when neuron j fires a spike. for t > 0, we see in the graph below the response/change in ![u_i(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29) which is a positive spike.

When the voltage difference between rest and moment t is positive (![u_i(t)-u_rest(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29%20-%20u_%7Brest%7D)), we call this an **Excitatory Postsynaptic Potential** or **EPSP**. When it is negative, we call this an **Inhibitory Postsynaptic Potential** or **IPSP**.

![Membrane Potential Example A](https://neuronaldynamics.epfl.ch/online/x6.png)

**FIG. 1**: We see the reaction inside of the membrane of the neuron. There is the Postsynaptic potential. Notice that this is not the Action Potential of the neuron. here we are showing the inputs of the postsynaptic neuron. If we were to reach _v_, then, we would have the triggering of an action potential in neuron i. _ESPS and ISPS tell us the type of input recieved by a neuron_

_Definition of Postsynaptic Potential_

![u_{i}(t) - u_{rest} =: \epsilon _{ij}(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29%20-%20u_%7Brest%7D%20%3D%3A%20%5Cepsilon%20_%7Bij%7D%28t%29)

Where Epsilon is the Postsynaptic potential by definition of the difference in u_t and u_rest. You can see this as being a formula for the postsynaptic potential at time t. For example, we see in figure 1 that at time t=0, we register the spike comming from neuron j in the membrane of the Soma/Dendrite of neuron i.

We can define the time course of a specific EPSP caused by a spike from neuron j in neuron i as:

![potential](https://neuronaldynamics.epfl.ch/online/x6.png)
![\epsilon _{ij}(t - t^{(f)}_{j})](https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bij%7D%28t%20-%20t%5E%7B%28f%29%7D_%7Bj%7D%29)

Which mathematically describes what is going on between the time of spike f from neuron j in the membrane of neuron i and some time t where t > t^(f)_j > 0

![The accumulation of EPSPs](https://neuronaldynamics.epfl.ch/online/x9.png)

**FIG. 2**: We can see the accumulation of Postsynaptic potentials in the membrane of the soma of neuron i when we recieve signals from neurons j=1, 2 (2 neurons). We never reach the threshold _v_.

![Reaching of the threshold](https://neuronaldynamics.epfl.ch/online/x10.png)
![reaching potential](https://neuronaldynamics.epfl.ch/online/x9.png)

**FIG. 3**: We see that after recieving multiple EPSPs, the PSP builds up towards threshold _v_ until we reach the threshold around ![\epsilon _{ij}(t - t^{(2)}_{1})](https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bij%7D%28t%20-%20t%5E%7B%282%29%7D_%7B1%7D%29)