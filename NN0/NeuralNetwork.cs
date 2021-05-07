using NN0.Functions;
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
        public NeuralNetwork(double descentStep = 0.01)
        {
            DescentStep = descentStep;
        }
        public double DescentStep { get; set; }
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
                Thread.Sleep(0);

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
        public void LearnWithSelection(Selection epoch, int times)
        {
            for (var i = 0; i < times; i++)
            {
                Console.WriteLine($"current lesson is {i}");
                while (epoch.HasNextRandomSample)
                {
                    var sample = epoch.GetNextRandomSample();
                    Learn(sample.InputVector, sample.AwaitedResponse);
                }
                epoch.ResetRandomizer();
            }
        }
        public void CheckWithSelection(Selection selection) 
        {
            foreach (var sample in selection.Samples)
            {
                var result = Calculate(sample.InputVector);
                DisplayResult(result);
            }
        }
        public void CheckWithInputVector(IEnumerable<double> inputVector)
        {
            var res = Calculate(inputVector);
            DisplayResult(res);

        }
        public void CheckWithMultipleVectors(IEnumerable<IEnumerable<double>> inputVectors)
        {
            foreach (var inputVector in inputVectors)
                CheckWithInputVector(inputVector);
        }
        private void DisplayResult(IEnumerable<double> result)
        {
            var shortResult = result.Select(r => Math.Round(r, 3));
            Console.WriteLine($"result: {string.Join(", ", shortResult)}");
            Console.WriteLine();
        }
            
        private void BackPropagate(Neuron neuron, double awaitedValue)
        {
            var outputValue = neuron.OutputValue;
            var error = outputValue - awaitedValue;

            var localGradient = error * neuron.ActivationFunction.Derivative(outputValue);
            //Console.WriteLine($"LastLayer error = {error}, local gradient = {localGradient}");
            var synapsesToModify = neuron.Dendrites.ToList();
            if (neuron.SynapseToBias != null)
                synapsesToModify.Add(neuron.SynapseToBias);

            foreach (var d in synapsesToModify)
            {
                var previousLayerNeuron = d.GetOtherNeuron(neuron);
                var previousOutputValue = previousLayerNeuron.OutputValue;
                var weight = d.Weight;
                d.Weight = weight - DescentStep * localGradient * previousOutputValue;

                previousLayerNeuron.BackPropagate(d, localGradient, DescentStep);
            };
        }
    }
    public class NeuralNetworkFactory
    {
        public static readonly double DEFAULT_WEIGHT = 1;
        public static NeuralNetwork CreateByFunctionTypeAndLayerSizes(ActivationFunctionType type,
             IEnumerable<int> layerSizes, double descentStep = 0.01)
        {
            var function = FunctionFactory.GetByType(type);
            return CreateByFunctionAndLayerSizes(function, layerSizes, descentStep);
        }
        public static NeuralNetwork CreateByFunctionAndLayerSizes(IActivationFunction activationFunction, 
            IEnumerable<int> layerSizes, double descentStep = 0.01) 
        {
            var network = new NeuralNetwork(descentStep);
            var networkNeurons = new List<Neuron>();
            var layersCount = layerSizes.Count();
            var previousLayer = new List<Neuron>();

            for (int i = 0; i < layersCount; i++)
            {
                var layerSize = layerSizes.ElementAt(i);
                if (layerSize < 1)
                    continue;

                // Creating neurons for the layer includind one extra neuron to be the bias
                var neuronsOnLayer = Enumerable.Range(1, layerSize).Select(o => new Neuron(activationFunction))
                    .ToList();
                networkNeurons.AddRange(neuronsOnLayer);

                // If it's the first layer
                if (i == 0)
                {
                    neuronsOnLayer.ToList().ForEach(n => n.IsOnTheFirstLayer = true);
                    network.InputNeurons = neuronsOnLayer.ToList();
                    previousLayer = neuronsOnLayer;
                    AddBias(networkNeurons, previousLayer, activationFunction);
                }

                // If it's not the first nor the last layer
                else if (i < layersCount - 1)
                {
                    SubscribeOneLayerToAnother(neuronsOnLayer, previousLayer);
                    previousLayer = neuronsOnLayer;
                    AddBias(networkNeurons, previousLayer, activationFunction);
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
        // Add a bias to the current layer
        private static void AddBias(
            IList<Neuron> networkNeurons, IList<Neuron> previousLayer, IActivationFunction function)
        {
            var biasOnLayer = new Neuron(function) { IsBias = true };
            networkNeurons.Add(biasOnLayer);
            previousLayer.Add(biasOnLayer);
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
