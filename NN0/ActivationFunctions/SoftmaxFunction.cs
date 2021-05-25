using NN0.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class SoftmaxFunction : IActivationFunction
    {
        private static List<Neuron> _neuronsWithASum = new List<Neuron>();
        private NeuralNetwork _currentNeuralNetwork;
        private Neuron _currentNeuron;
        
        public SoftmaxFunction(NeuralNetwork nn, Neuron currentNeuron)
        {
            _currentNeuralNetwork = nn;
            _currentNeuron = currentNeuron;
        }
        
        public void ApplySum()
        {
            _neuronsWithASum.Add(_currentNeuron);
            // If not all neurons has a sum yet
            if (!_currentNeuralNetwork.OutputNeurons.All(n => _neuronsWithASum.Contains(n)))
                return;

            // Else, when all neurons asked an output and has a sum
            // calculate an output value for each neuron
            var e = Math.E;

            foreach (var neuron in _neuronsWithASum)
            {
                var localSum = neuron.Sum;
                var output = Math.Pow(e, localSum) / _neuronsWithASum.Sum(n => Math.Pow(e, n.Sum));

                neuron.OutputValue = output;
                neuron.IsCalculationComplete = true;
                neuron.SendSignal();
            }
            _neuronsWithASum.Clear();
        }

        public double SoftmaxDerivative(double outputValue, double expectedValue)
        {
            return outputValue - expectedValue;
        }
    }
}
