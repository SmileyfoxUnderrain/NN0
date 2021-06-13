using NN0.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0
{
    public class NeuralNetworkFactory
    {
        public static readonly double DEFAULT_WEIGHT = 1;
        private NeuralNetwork _nn;
        private IList<Layer> _layers;

        public static NeuralNetwork CreateByFunctionTypeAndLayerSizes(ActivationFunctionType type,
             IEnumerable<int> layerSizes, double lambda = 0.01)
        {
            var function = ActivationFunctionFactory.GetByType(type);
            return CreateByFunctionAndLayerSizes(function, layerSizes, lambda);
        }
        public NeuralNetworkFactory PrepareNetwork(double lambda = 0.01)
        {
            _layers = new List<Layer>();
            _nn = new NeuralNetwork(lambda);
            return this;
        }
        public NeuralNetworkFactory SetInputLayer(int inputsCount, bool hasBias = true)
        {
            if (inputsCount < 1)
                throw new ArgumentException("Input count can't be less than one");
            if (_nn.InputNeurons != null
                && _nn.InputNeurons.Any() 
                && !_nn.Neurons.All(n => _nn.InputNeurons.Contains(n)))
                throw new NotImplementedException(
                    "Changing an input layer when have more neurons not implemented yet");

            var neuronsOnLayer = Enumerable.Range(1, inputsCount).Select(o => new Neuron(null, false, true))
                   .ToList();
            _nn.InputNeurons = neuronsOnLayer.ToList();
            if (hasBias)
            {
                var bias = new Neuron(null, true, true);
                neuronsOnLayer.Add(bias);
            }
            _nn.Neurons = neuronsOnLayer.ToList();
            var layer = new Layer(neuronsOnLayer, 0);
            _layers.Add(layer);
            return this;
        }
        public NeuralNetworkFactory AddLayer(
            IActivationFunction activationFunction, int neuronsCount, bool hasBias, bool isOutputLayer = false)
        {
            if (_layers == null || !_layers.Any() || !_layers.Any(l => l.LayerNumber == 0)
                || _nn.InputNeurons == null || !_nn.InputNeurons.Any())
                throw new NotImplementedException("Creating any layer without input layer not implemented yet");
            if (_nn.OutputNeurons!=null && _nn.OutputNeurons.Any())
                throw new NotImplementedException(
                    "Adding a layer when output layer already added, not implemented yet");

            var neuronsOnLayer = Enumerable.Range(1, neuronsCount).Select(o => new Neuron(activationFunction))
                .ToList();

            var previousLayer = _layers.OrderBy(l => l.LayerNumber).Last();
            SubscribeOneLayerToAnother(neuronsOnLayer, previousLayer.Neurons);

            if (isOutputLayer)
                _nn.OutputNeurons = neuronsOnLayer.ToList();
            else if (hasBias)
            {
                var bias = new Neuron(null, true, false);
                neuronsOnLayer.Add(bias);
            }
            var layer = new Layer(neuronsOnLayer, previousLayer.LayerNumber + 1);
            _layers.Add(layer);
            _nn.Neurons = _nn.Neurons.Concat(neuronsOnLayer);
            return this;
        }
        public NeuralNetworkFactory AddLayer(
            ActivationFunctionType activationFunctionType, int neuronsCount, bool hasBias, bool isOutputLayer = false)
        {

            if (activationFunctionType != ActivationFunctionType.Softmax)
            {
                var activationFunction = ActivationFunctionFactory.GetByType(activationFunctionType);
                return AddLayer(activationFunction, neuronsCount, hasBias, isOutputLayer);
            }
            else
            {
                if (!isOutputLayer)
                    throw new ArgumentException("Can't apply a softmax function for the non-output layer");
                AddLayer(null, neuronsCount, hasBias, true);
                foreach (var neuron in this._nn.OutputNeurons)
                    neuron.ActivationFunction = new SoftmaxFunction(_nn, neuron);
                return this;
            }   
        }
        public NeuralNetwork GetPreparedNetwork()
        {
            _nn.Neurons.ToList().ForEach(n => n.CurrentNeuralNetwork = _nn);
            if (_nn.InputNeurons != null && _nn.InputNeurons.Any() 
                && _nn.OutputNeurons != null && _nn.OutputNeurons.Any())
                return _nn;

            throw new InvalidOperationException("Neural network must have input and output layers");
        }

        public static NeuralNetwork CreateByFunctionAndLayerSizes(IActivationFunction activationFunction,
            IEnumerable<int> layerSizes, double lambda = 0.01)
        {
            var network = new NeuralNetwork(lambda);
            var networkNeurons = new List<Neuron>();
            var layersCount = layerSizes.Count();
            var previousLayer = new List<Neuron>();

            for (int i = 0; i < layersCount; i++)
            {
                var layerSize = layerSizes.ElementAt(i);
                if (layerSize < 1)
                    continue;

                // Creating neurons for the layer
                var neuronsOnLayer = Enumerable.Range(1, layerSize).Select(o => new Neuron(activationFunction))
                    .ToList();
                networkNeurons.AddRange(neuronsOnLayer);

                // If it's the first layer
                if (i == 0)
                {
                    neuronsOnLayer.ToList().ForEach(n => n.IsOnTheFirstLayer = true);
                    network.InputNeurons = neuronsOnLayer.ToList();
                    previousLayer = neuronsOnLayer;
                    AddBias(networkNeurons, previousLayer);
                }

                // If it's not the first nor the last layer
                else if (i < layersCount - 1)
                {
                    SubscribeOneLayerToAnother(neuronsOnLayer, previousLayer);
                    previousLayer = neuronsOnLayer;
                    AddBias(networkNeurons, previousLayer);
                }
                // The output layer
                else
                {
                    SubscribeOneLayerToAnother(neuronsOnLayer, previousLayer);
                    network.OutputNeurons = neuronsOnLayer;
                }
            }
            network.Neurons = networkNeurons;
            network.Neurons.ToList().ForEach(n => n.CurrentNeuralNetwork = network);
            return network;
        }
        // Add a bias to the current layer
        private static void AddBias(
            IList<Neuron> networkNeurons, IList<Neuron> previousLayer)
        {
            var biasOnLayer = new Neuron(null, true, false);
            networkNeurons.Add(biasOnLayer);
            previousLayer.Add(biasOnLayer);
        }

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
                }
        }
    }
    public class Layer
    {
        public Layer(IEnumerable<Neuron> neurons, int layerNumber)
        {
            Neurons = neurons;
            LayerNumber = layerNumber;
        }

        public IEnumerable<Neuron> Neurons { get; set; }
        public int LayerNumber { get; set; }

    }
}
