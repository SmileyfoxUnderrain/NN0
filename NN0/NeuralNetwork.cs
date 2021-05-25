using NN0.ActivationFunctions;
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
            //Console.WriteLine($"Received:   {string.Join(", ", inputs.Select(i => i.ToString("F")))}");
            //Console.WriteLine($"Calculated: {string.Join(", ", outputValues.Select(i => i.ToString("F")))}");
            //Console.WriteLine($"Awaited:    {string.Join(", ", outputs.Select(i => i.ToString("F")))}");

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
            double localGradient;

            if (neuron.ActivationFunction is SoftmaxFunction neuronSoftMax)
                localGradient = neuronSoftMax.SoftmaxDerivative(outputValue, expectedValue);
            else if (neuron.ActivationFunction is ILayerIndependentFunction neuronFunction)
                localGradient = error * neuronFunction.Derivative(outputValue);
            else
                throw new NotImplementedException(
                    $"Back propagation for the {neuron.ActivationFunction.GetType().Name} type not implemented yet");
            
            //Console.WriteLine($"LastLayer error = {error}, local gradient = {localGradient}");
            var synapsesToModify = neuron.Dendrites.ToList();
            if (neuron.SynapseToBias != null)
                synapsesToModify.Add(neuron.SynapseToBias);

            foreach (var d in synapsesToModify)
            {
                //Console.WriteLine($"Backpropagate for synapse {synapsesToModify.LastIndexOf(d)}");
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
    
}
