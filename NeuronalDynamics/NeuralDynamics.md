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

The reason that we do t-t^f_j is that if we want to see the time course for a single spike, it makes sense if we "translated" the graph for the spike back to zero. This would allow us to not have to move to time t^f_j!

When the voltage difference between rest and moment t is positive (![u_i(t)-u_rest(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29%20-%20u_%7Brest%7D)), we call this an **Excitatory Postsynaptic Potential** or **EPSP**. When it is negative, we call this an **Inhibitory Postsynaptic Potential** or **IPSP**.

![Membrane Potential Example A](https://neuronaldynamics.epfl.ch/online/x6.png)

**FIG. 1**: We see the reaction inside of the membrane of the neuron. There is the Postsynaptic potential. Notice that this is not the Action Potential of the neuron. here we are showing the inputs of the postsynaptic neuron. If we were to reach _v_, then, we would have the triggering of an action potential in neuron i. _ESPS and ISPS tell us the type of input recieved by a neuron_

_Definition of Postsynaptic Potential_

![u_{i}(t) - u_{rest} =: \epsilon _{ij}(t)](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29%20-%20u_%7Brest%7D%20%3D%3A%20%5Cepsilon%20_%7Bij%7D%28t%29)

Where Epsilon is the Postsynaptic potential by definition of the difference in u_t and u_rest. You can see this as being a formula for the postsynaptic potential at time t. For example, we see in figure 1 that at time t=0, we register the spike comming from neuron j in the membrane of the Soma/Dendrite of neuron i.

We can define the time course of a specific EPSP caused by a spike from neuron j in neuron i as:

![\epsilon _{ij}(t - t^{(f)}_{j})](https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bij%7D%28t%20-%20t%5E%7B%28f%29%7D_%7Bj%7D%29)

Which mathematically describes what is going on between the time of spike f from neuron j in the membrane of neuron i and some time t where t > t^(f)_j > 0.
This is why we treat epsilon _as some kind of function, but it has a more special mathematical meaning that just a function._ __Epsilon represents the PSP events.__ You could also see Epsilon as the function for a specific spike that returns the maximum voltage reached by the PSP event. We can also find the _Current spike_. We denote this as t - t^(f)_j meaning the current spike level at time t for event f from neuron j in membrane of neuron i.

![potential](https://neuronaldynamics.epfl.ch/online/x6.png)
![The accumulation of EPSPs](https://neuronaldynamics.epfl.ch/online/x7.png)

**FIG. 2**: We can see the accumulation of Postsynaptic potentials in the membrane of the soma of neuron i when we recieve signals from neurons j=1, 2 (2 neurons). We never reach the threshold _v_.

![Reaching of the threshold](https://neuronaldynamics.epfl.ch/online/x8.png)
![reaching potential](https://neuronaldynamics.epfl.ch/online/x9.png)

**FIG. 3**: We see that after recieving multiple EPSPs, the PSP builds up towards threshold _v_ until we reach the threshold around ![\epsilon _{ij}(t - t^{(2)}_{1})](https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bij%7D%28t%20-%20t%5E%7B%282%29%7D_%7B1%7D%29)

We can calculate the total change in the potential of the membrane by adding the individual PSPs if and only if _we do not reach the threshold level_.

![Total change of potential](https://latex.codecogs.com/gif.latex?u_%7Bi%7D%28t%29%3D%5Csum_%7Bj%7D%20%5Csum_%7Bf%7D%5Cepsilon%20_%7Bij%7D%28t-t%5E%7Bf%7D_%7Bj%7D%29%20&plus;%20u_%7Brest%7D%20%5CRightarrow%20u_%7Bi%7D%28t%29%20%3C%20%5Cvartheta)

We can see that we can simply add the Postsynaptic potentials at time t for each neuron and each spike (if a spike was triggered way before t, it should be 0 basically) and add to it the resting level to find the total Postsynaptic potential at time t. _Notice that the summation part of the equation give us the change (or better said, the amount above resting lever) in action potential at time t_. If we look at figure 4, we can see how at 
![EPSP](https://latex.codecogs.com/gif.latex?%5Cepsilon%20_%7Bi2%7D%28t-t%5E%7B%282%29%7D_%7B2%7D%29) 
The membrane reaches the threshold level after which the linearity of the neuron goes away and instead follows some very specific behaviour:
1. membrane potential echibits pulse-like excursion reaching 100mV (Presynaptic levels) and _propagates along the axon of neuron i._
2. Potential experiences a period of _hyperpolarization_ (aka. at negative levels.) also called __Spike-afterpotential__.

![threshold](https://neuronaldynamics.epfl.ch/online/x10.png)
![threshold](https://neuronaldynamics.epfl.ch/online/x11.png)

**FIG. 4**: We can see the reaching of the threshold levels and the spike-afterpotential period after the second spike comming from neuron 2.

**It is the combination of Postsynaptic Potentials that will cause the neuron to have an action potential. As we can see, the strength of the PSP of each neuron is the exact same, but the timing and frequency cause the Postsynaptic neuron's membrane potential to raise (or fall if dealing with IPSP) until reaching a threshold level.**

### Integrate-and-fire models

There exist many models that model the way that neurons react and act in a mathematical way. One of the most basic, and the base for other models, is the *Integrate-and-fire models*. These models consist of 2 parts:
1. A *linear* differential equation to describe the evolution/change in membrane potential.
2. A threshold for spike firing.

An important idea to take away is that neurons are seen as summation processes: We have to add the Postsynaptic potentials to check if we reach a threshold level v. the moment that we cross that threshold is refered to as ![t_i^f](https://latex.codecogs.com/gif.latex?t_%7Bi%7D%5E%7B%28f%29%7D). Notice how we refer to the postsynaptic, rather than the presynaptic neuron in the index of t. Once the neuron reaches the ![threshold](https://latex.codecogs.com/gif.latex?t_%7Bi%7D%5E%7B%28f%29%7D) point, _we use a mechanism to generate spikes_.

We do not care for the shape of the spikes: they dont give information. It is their timing and sequences that matter.

#### The Leaky Integrate-and-Fire Model

Many models build on top of this model. It is based on the notion that a neuron can be described as a circuit where we have a resistor and a capacitor in parallel. we have a constant resting voltage ![u_rest](https://latex.codecogs.com/gif.latex?u_%7Brest%7D) that keeps a minimum voltage in the cell. We then have I(t) which is an external voltage applied to the system that raises the voltage in the system for a period of time before decaying.

![](https://neuronaldynamics.epfl.ch/online/x12.png)
![](https://neuronaldynamics.epfl.ch/online/x13.png)

**FIG. 6**: Notice how the capacitor in the circuit represents the cell membrane that creates a type of capacitor, since the membrane is a pretty good insulator. As all things in real life, the insulation is not perfect, therefore it "leaks" over time. This is the escence of leaky Integrate-and-fire. The resistor in the circuit creates a delay time for the "leaking". **We refer to the action potentials as events.**

##### Leak Integrator Formula/ Passive membrane

Assume that at time t=0, the membrane potential is ![](https://latex.codecogs.com/gif.latex?u_%7Brest%7D%20&plus;%20%5CDelta%20u). What this means is that at time 0, we expect a spike to have happened. for t > 0, input current vanishes I(t) and the membrane potential goes down to its rest level.

![](https://latex.codecogs.com/gif.latex?u%28t%29-u_%7Brest%7D%20%3D%20%5CDelta%20u%20%5C%3A%20exp%28-%20%5Cfrac%7Bt-t_%7B0%7D%7D%7B%5Ctau%20_%7Bm%7D%7D%29%20%5C%3B%5C%3B%20for%20%5C%3B%5C%3B%20t%20%3E%20t_%7B0%7D)

where ![](https://latex.codecogs.com/gif.latex?%7B%5Ctau%20_%7Bm%7D%7D) = RC which is the characteristic time of the decay for the membrane potential. 

![](https://latex.codecogs.com/gif.latex?%5CDelta%20u) is the level the spike reaches. In real life it is around 1 mV.

This equation tell us that after an event is gone, the membrane potential decays in an exponential fashion. the delay of this decay is dependant on tau (in neurons is around 10ms).

##### The Pulse Input

###### The Dynamics of the Pulse Input

By integrating the *Leaky Integrator* formula, we can analyze the nature of the pulses as postsynaptic potentials.

The pulse input that comes from the presynaptic neuron does not transmit any information in its amplitude, but rather on the timing of the spikes generated and their rate.

![](https://latex.codecogs.com/gif.latex?u%28t%29-u_%7Brest%7D%20%3D%20%5CDelta%20u%20%5C%3B%20exp%28-%5Cfrac%7Bt-t_%7B0%7D%7D%7B%5Ctau%20_%7Bm%7D%7D%29)

becomes the differential equation

![](https://latex.codecogs.com/gif.latex?%5Ctau%20_%7Bm%7D%20%5Cfrac%7B%5Cmathrm%7Bd%7Du%20%7D%7B%5Cmathrm%7Bd%7D%20t%7D%20%3D%20-%5Bu%28t%29%20-%20u_%7Brest%7D%5D%20&plus;%20R%5C%3AI%28t%29)

After integrating, we obtain

![](https://latex.codecogs.com/gif.latex?u%28t%29%3Du_%7Brest%7D&plus;R%5C%3AI_%7B0%7D%5B1-exp%28-%5Cfrac%7Bt%7D%7B%5Ctau_%7Bm%7D%7D%29%5D)

We asume that a spike comes in at time t = ![delta](https://latex.codecogs.com/gif.latex?%5CDelta). As our initial potential, we have U_rest. we integrate for 0 < t < ![delta](https://latex.codecogs.com/gif.latex?%5CDelta)

We can use the dirac delta function for our analysis.

_The dirac delta function is the function ![](https://latex.codecogs.com/gif.latex?%5Cdelta%28x%29%20%3D%200%20%5C%3B%20for%5C%3Bx%5Cneq%200) and its integral equals 1. The graph of the function is similar to a large plane with a stick standing in the middle of it. It is a mathematical abstraction but helps when analyzing equations/systems where we may need to approach some value ![](https://latex.codecogs.com/gif.latex?%5Cpsi) as we approach some value (delta or t in our case) to 0 as in ![](https://latex.codecogs.com/gif.latex?%5Clim_%7B%5CDelta%5Crightarrow%200%7D%5Cfrac%7B%5Cpsi%20%7D%7B%5CDelta%7D%20%3D%20%5Cpsi%5Clim_%7B%5CDelta%5Crightarrow%200%7D%5Cfrac%7B1%20%7D%7B%5CDelta%7D)._

In the case of our analysis, we use the dirac delta function to model the spike behaviour of an input event.

If we look at the resulting integral of the leaky integrator, we see that we depend on the variable t which would be the time course of out spike. This time course is actually defined as ![](https://latex.codecogs.com/gif.latex?0%20%3C%20t%20%3C%20%5CDelta) where uppercase delta is the duration of time the spike has. It would be ideal to modelate a spike as what it is: (some "bump" at a finite moment in time). We can accomplish this by taking the limit of the function as we approach 0 for delta.

Now, based on out previous observation that the shape of a spike is always the exact same, we can also infer that the total charge of a spike is some fixed constant q. This can be modeled as the integral of the spike I(t), where I, as in the circuit diagram represents, is the incomming spike. we know that this spike is always the value q. we can use the dirac function to now shorten the length the spikes takes to near 0.

![](https://neuronaldynamics.epfl.ch/online/x14.png)

as we make the pulse "slimmer" (above), we notice how the PSP changes to become more of a triangle shape. It is important to note, once again, that the area under the curves always remain the same. we can model this with the equation 

![](https://latex.codecogs.com/gif.latex?I%28t%29%20%3D%20q%5Cdelta%20%28t%29)

The dirac function takes off of our backs the time course t (the membrane potential just jumps at time t=0)

##### The Threshold for Spike Firing

We refer to the moment when a spike "fires" or emits its action potential as the firing time and is described mathematically as the firing time ![](https://latex.codecogs.com/gif.latex?t%5E%7B%28f%29%7D). This event happens in the leaky integrate-and-fire when the postsynaptic potential passes the threshold v and is described as:

![](https://latex.codecogs.com/gif.latex?t%5E%7B%28f%29%7D%3A%5C%3Bu%28t%5E%7B%28f%29%7D%29%20%3D%20%5Cvartheta)

Once more, the form of a spike is rather irrelevant, but the time of firing is.
At time ![](https://latex.codecogs.com/gif.latex?t%5E%7B%28f%29%7D) an action potential is sents/created but right after it what happens? We know that the linearity of the system breaks down when we cross the threshold, so a very specific set of actions happen. The potential comes down to some value ![](https://latex.codecogs.com/gif.latex?u_%7Br%7D). Again, because we model the spike as literally some spike at the exact moment ![](https://latex.codecogs.com/gif.latex?t%5E%7B%28f%29%7D) where its length of time approaches 0 we define the "reset" period with the following equation:

![](https://latex.codecogs.com/gif.latex?%5Clim_%7B%5Cdelta%5Crightarrow0%3B%5Cdelta%3E0%7D%5C%3Au%28t%5E%7B%28f%29%7D&plus;%5Cdelta%29%20%3D%20u_%7Br%7D)

Notice how this limit is a right-hand-sided limit. We already know that at some point after the spike time the potential decays back to u_rest, but in between the potential decays to some value u_r which is the reset potential. Now, when does this happen? well, this happens immediately after the spike occurs at its time t^(f). The limit "pushes" the time of this occurence as close as possible as "right after" the spike. **After that moment, the neuron falls back to u_rest and the behaviour of the neuron is given by the leky fire-and-integrate equation.**

**The combination of the two mentioned equations (leaky integrate-and-fire and spike-firing equations) create the _Leaky Integrate-and-Fire_**

![](https://neuronaldynamics.epfl.ch/online/x16.png)

**FIG 7.** Example of an Integrate-and-Fire model. Here we appreciate the time course when we have a constant input current of 1.5 (this is what makes the curves so smooth compared with FIG 8 which is time dependent.) When ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5CDelta%20u%28t%29%20/%20%5Cvartheta) >= 1, the threshold values has been crossed, and we move from following the behaviour in the leaky integrator equation to the one of spike-firing. Right after the time t, the potential is reset to u_r which is the same thing as u_rest

