using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0
{
    public class Synapse
    {
        public Synapse(Neuron neuronA, Neuron neuronB, double weight)
        {
            NeuronA = neuronA;
            NeuronB = neuronB;
            Weight = weight;
        }

        public Neuron NeuronA {get; set;}
        public Neuron NeuronB {get; set;}
        public double Weight { get; set; }
        public Neuron GetOtherNeuron(Neuron currentNeuron)
        {
            if (NeuronA == currentNeuron)
                return NeuronB;
            if (NeuronB == currentNeuron)
                return NeuronA;
            throw new ArgumentException("The connection does not belongs to the given neuron");
        }
    }
}
