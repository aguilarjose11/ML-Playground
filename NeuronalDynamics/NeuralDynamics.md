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

![Membrane Potential Example](https://neuronaldynamics.epfl.ch/online/x6.png)

An input at an excitatory synapse that reduces polatization (since cell mebrane is naturally negativelly charged ~-65mV) it is called _depolarizing_ (V > -65mV), if it increases the negativity, it is called _hyperpolarizing_.

**Postsynaptic Potential (PSP) formula**
for a time course (change in membrane potential) u, in the neuron i, we denote this the change in membrane potetial in neuron *i* as ![u_i(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29) 