using NN0.Functions;
using NN0.LossFunctions;
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
        public NeuralNetwork(double lambda = 0.01)
        {
            Lambda = lambda;
        }
        /// <summary>
        /// Regularization rate for the descent steps
        /// </summary>
        public double Lambda { get; set; }
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

        public void Train(IEnumerable<double> inputs, IEnumerable<double> outputs)
        {
            
            if (outputs.Count() != OutputNeurons.Count())
                throw new ArgumentException(
                    $"Incorrect number of output values. Given is {outputs.Count()} but {OutputNeurons.Count()} expected");

            Calculate(inputs);

            for (var i = 0; i < OutputNeurons.Count(); i++)
                BackPropagate(OutputNeurons.ElementAt(i), outputs.ElementAt(i));

        }
        public void TrainWithSelection(Selection sel, int times)
        {
            for (var i = 0; i < times; i++)
            {
                Console.WriteLine($"current lesson is {i}");
                while (sel.HasNextRandomSample)
                {
                    var sample = sel.GetNextRandomSample();
                    Train(sample.InputVector, sample.ExpectedResponse);
                }
                sel.ResetRandomizer();
            }
        }
        public void TrainWitLossFunctionAndDifferentSets(ILossFunction lossFunction, int maxTimes, Selection sel, Selection validationSet)
        {
            for (var i = 0; i < maxTimes; i++)
            {
                // Teach the nn with training data set
                Console.Write($"current lesson is {i}");
                while (sel.HasNextRandomSample)
                {
                    var sample = sel.GetNextRandomSample();
                    Train(sample.InputVector, sample.ExpectedResponse);
                }
                sel.ResetRandomizer();

                // Evaluate the total loss for the trainig data set
                double totalLoss = 0;
                while (sel.HasNextRandomSample)
                {
                    var sample = sel.GetNextRandomSample();
                    var loss = CalculateLossForSample(sample, lossFunction);
                    totalLoss += loss;
                }
                sel.ResetRandomizer();
                var averageLoss = totalLoss / sel.Samples.Count();

                // Evaluate the loss for validation data set
                var validationLoss =
                    validationSet.Samples.Sum(s => CalculateLossForSample(s, lossFunction)) / validationSet.Samples.Count;

                var lossDelta = Math.Abs(validationLoss - averageLoss);

                Console.WriteLine($" Teacing loss = {averageLoss}, Validation loss = {validationLoss}, delta = {lossDelta}");
            }
        }
        public void TrainWithLossFunctionUsingCrossValidation(ILossFunction lossFunction, int maxTimes, Selection sel)
        {
            if (sel.Samples.Count() <= 1)
                throw new ArgumentException("Not enough samples to perform training");

            for(var i = 0; i < maxTimes; i++)
            {
                // On the current epoch
                // 20% of samples will be used to control 
                // and will not participate in training directly
                // The selection of the validation sample happens each training cycle
                // It's called a cross-validation
                int validationSetSize = sel.Samples.Count() / 5;
                var validationSet = new List<Sample>();
                for(int v = 0; v < validationSetSize; v++)
                {
                    validationSet.Add(sel.GetNextRandomSample());
                }

                // Teach the nn with training data set
                Console.Write($"current lesson is {i}");
                while (sel.HasNextRandomSample)
                {
                    var sample = sel.GetNextRandomSample();
                    Train(sample.InputVector, sample.ExpectedResponse);
                }
                sel.ResetRandomizer();

                // Evaluate the total loss for the trainig data set
                double totalLoss = 0;
                while (sel.HasNextRandomSample)
                {
                    var sample = sel.GetNextRandomSample();
                    if (validationSet.Contains(sample))
                        continue;

                    var loss = CalculateLossForSample(sample, lossFunction);
                    totalLoss += loss;
                }
                sel.ResetRandomizer();
                var averageLoss = totalLoss / sel.Samples.Count();

                // Evaluate the loss for validation data set
                var validationLoss =
                    validationSet.Sum(s => CalculateLossForSample(s, lossFunction)) / validationSetSize;

                var lossDelta = Math.Abs(validationLoss - averageLoss);
                
                Console.WriteLine($" Teacing loss = {averageLoss}, Validation loss = {validationLoss}, delta = {lossDelta}");
            }
        }
        public void TrainWithLossType(LossFunctionType lossType, int maxTimes, Selection sel, Selection validationSet = null)
        {
            var lossFunction = LossFunctionFactory.GetByType(lossType);
            if (validationSet == null)
                TrainWithLossFunctionUsingCrossValidation(lossFunction, maxTimes, sel);
            else
                TrainWitLossFunctionAndDifferentSets(lossFunction, maxTimes, sel, validationSet);
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
        public void SetOutputActivationFunctionType(ActivationFunctionType type)
        {
            var function = ActivationFunctionFactory.GetByType(type);
            SetOutputActivationFunction(function);
        }
        public void SetOutputActivationFunction(IActivationFunction activationFunction)
        {
            foreach(var neuron in OutputNeurons)
                neuron.ActivationFunction = activationFunction;
        }
            
        private void BackPropagate(Neuron neuron, double expectedValue)
        {
            var outputValue = neuron.OutputValue;
            var error = outputValue - expectedValue; //e = y - d

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
                d.Weight = weight - Lambda * localGradient * previousOutputValue;

                previousLayerNeuron.BackPropagate(d, localGradient, Lambda);
            };
        }
        private double CalculateLossForSample(Sample sample, ILossFunction lossFunction)
        {
            var result = Calculate(sample.InputVector);
            var receivedToExpectedPairs = result.Zip(sample.ExpectedResponse)
                .Select(z => new Tuple<double, double>(z.First, z.Second));
            var loss = lossFunction.CalculateLoss(receivedToExpectedPairs);
            return loss;
        }
    }
    public class NeuralNetworkFactory
    {
        public static readonly double DEFAULT_WEIGHT = 1;
        public static NeuralNetwork CreateByFunctionTypeAndLayerSizes(ActivationFunctionType type,
             IEnumerable<int> layerSizes, double lambda = 0.01)
        {
            var function = ActivationFunctionFactory.GetByType(type);
            return CreateByFunctionAndLayerSizes(function, layerSizes, lambda);
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
        // Assume that the weights are all equals the same value to be changed during training
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
