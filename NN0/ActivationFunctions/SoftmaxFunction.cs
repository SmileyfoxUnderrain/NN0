using NN0.ActivationFunctions;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class SoftmaxFunction : IActivationFunction
    {
        private static readonly ConcurrentBag<Neuron> _neuronsWithSumsConcurrent = 
            new ConcurrentBag<Neuron>();

        private static readonly object _lockObject = new();
        private static bool _outputLock;

        private NeuralNetwork _currentNeuralNetwork;
        private Neuron _currentNeuron;
        
        public SoftmaxFunction(NeuralNetwork nn, Neuron currentNeuron)
        {
            _currentNeuralNetwork = nn;
            _currentNeuron = currentNeuron;
        }
        
        public void ApplySum()
        {
            _neuronsWithSumsConcurrent.Add(_currentNeuron);
            // If not all neurons has a sum yet
            if (!_currentNeuralNetwork.OutputNeurons.All(n => _neuronsWithSumsConcurrent.Contains(n)))
                return;

            lock (_lockObject)
            {
                if (_outputLock)
                    return;

                _outputLock = true;
            }

            // Else, when all neurons asked an output and has a sum
            // calculate an output value for each neuron
            var e = Math.E;
            var sumOfExpsPowered = _neuronsWithSumsConcurrent.Sum(n => Math.Pow(e, n.Sum));
            foreach (var neuron in _neuronsWithSumsConcurrent)
            {
                var localSum = neuron.Sum;
                var output = Math.Pow(e, localSum) / sumOfExpsPowered;

                neuron.OutputValue = output;
                neuron.IsCalculationComplete = true;
            }
            _neuronsWithSumsConcurrent.Clear();
            _outputLock = false;
        }

        public double SoftmaxDerivative(double outputValue, double expectedValue)
        {
            return outputValue - expectedValue;
        }
    }
}
