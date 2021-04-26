using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NN0
{
    public class NeuralNetwork
    {
        private readonly double _step = 1;
        public IEnumerable<Neuron> InputNeurons { get; set; }
        public IEnumerable<Neuron> OutputNeurons { get; set; }
        public IEnumerable<Neuron> Neurons { get; set; }
        public IEnumerable<double> OutputValues
        {
            get
            {
                return OutputNeurons.Select(n => n.OutputValue);
            }
        }
        public void SendToInput(IEnumerable<double> inputSignals)
        {
            if (inputSignals.Count() != InputNeurons.Count())
                throw new ArgumentException(
                    $"Incorrect number of input values. Given is {inputSignals.Count()} but {InputNeurons.Count()} expected");

            ResetSums();

            for(var i = 0; i < InputNeurons.Count(); i++)
            {
                var value = inputSignals.ElementAt(i);
                var neuron = InputNeurons.ElementAt(i);
                neuron.ReceiveSignal(value);
            }
        }
        public void ResetSums()
        {
            Neurons.ToList().ForEach(n => n.Reset());
        }

        public IEnumerable<double> Calculate(IEnumerable<double> inputs)
        {
            Neurons.ToList().ForEach(n => n.Reset());
            Task.Run(() => SendToInput(inputs));
            // await for calculation completes
            while (!OutputNeurons.All(n => n.IsCalculationComplete))
                Thread.Sleep(5);

            return OutputValues;
        }

        public void Learn(IEnumerable<double> inputs, IEnumerable<double> outputs)
        {
            
            if (outputs.Count() != OutputNeurons.Count())
                throw new ArgumentException(
                    $"Incorrect number of output values. Given is {outputs.Count()} but {OutputNeurons.Count()} expected");

            Calculate(inputs);

            for (var i = 0; i < OutputNeurons.Count(); i++)
                BackPropagate(OutputNeurons.ElementAt(i), outputs.ElementAt(i));

        }
        private void BackPropagate(Neuron neuron, double awaitedValue)
        {
            var outputValue = neuron.OutputValue;
            var error = outputValue - awaitedValue;
            
            // sigma * omega * f * (1 - f)
            var localGradient = error * outputValue * (1 - outputValue);
            //Console.WriteLine($"LastLayer error = {error}, local gradient = {localGradient}");
            foreach (var d in neuron.Dendrites)
            {
                var previousLayerNeuron = d.GetOtherNeuron(neuron);
                var previousOutputValue = previousLayerNeuron.OutputValue;
                var weight = d.Weight;
                d.Weight = weight - _step * localGradient * previousOutputValue;

                previousLayerNeuron.BackPropagate(d, localGradient, _step);
            };
        }
    }
    public class NeuralNetworkFactory
    {
        public static readonly double DEFAULT_WEIGHT = 1;
        public static NeuralNetwork CreateByLayerSizes(IEnumerable<int> layerSizes) 
        {
            var network = new NeuralNetwork();
            var networkNeurons = new List<Neuron>();
            var layersCount = layerSizes.Count();
            var previousLayer = new List<Neuron>();

            for (int i = 0; i < layersCount; i++)
            {
                var layerSize = layerSizes.ElementAt(i);
                var neuronsOnLayer = Enumerable.Range(1, layerSize).Select(o => new Neuron())
                    .ToList();

                networkNeurons.AddRange(neuronsOnLayer);

                // If it's the first layer
                if (i == 0)
                {
                    neuronsOnLayer.ToList().ForEach(n => n.IsOnTheFirstLayer = true);
                    network.InputNeurons = neuronsOnLayer;
                    previousLayer = neuronsOnLayer;
                }

                // If it's not the firs nor the last layer
                else if (i < layersCount - 1)
                {
                    SubscribeOneLayerToAnother(neuronsOnLayer, previousLayer);
                    previousLayer = neuronsOnLayer;
                }
                // The output layer
                else
                {
                    SubscribeOneLayerToAnother(neuronsOnLayer, previousLayer);
                    network.OutputNeurons = neuronsOnLayer;
                }
            }
            network.Neurons = networkNeurons;
            return network;
        }
        // Subscribe every neuron on the current layer with everry neuron on the previous layer
        // Assume that the weights are all equals the same value to be changed during learning
        private static void SubscribeOneLayerToAnother(IEnumerable<Neuron> currentLayer, IEnumerable<Neuron> previousLayer)
        {
            var rnd = new Random(DateTime.Now.Ticks.GetHashCode());
            foreach (var currentNeuron in currentLayer)
                foreach (var prevNeuron in previousLayer)
                {
                    var randomWeight = DEFAULT_WEIGHT * rnd.NextDouble();
                    var connection = new Synapse(prevNeuron, currentNeuron, randomWeight);
                    prevNeuron.Synapses.Add(connection);
                    currentNeuron.Synapses.Add(connection);
                    prevNeuron.Signal += currentNeuron.OnIncomingSignal;
                }
        }

    }
}
