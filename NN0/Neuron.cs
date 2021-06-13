using System;
using System.Collections.Specialized;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using NN0.ActivationFunctions;
using System.Collections.Concurrent;

namespace NN0
{
    public class Neuron
    {
        private readonly ConcurrentBag<double> _summarizationBag = new ConcurrentBag<double>();
        private readonly ConcurrentDictionary<Synapse, double> _dendritesInputsConcurrent =
            new ConcurrentDictionary<Synapse, double>();

        private readonly ConcurrentDictionary<Synapse, double> _axonsOutputsConcurrent =
            new ConcurrentDictionary<Synapse, double>();

        private readonly object _lockObject = new object();
        private bool _sumLock;
        private bool _backPropagationLock;

        private const double BIAS_OUTPUT = 1;
        private bool _isOnTheFirstLayer;
        private bool _isBias;
        private double _sum;
        protected int _stepsToInput = int.MaxValue;

        // Neuron needs to be synchronized in the network
        public Neuron(IActivationFunction activationFunction, bool isBias = false, bool isOnTheFirstLayer = false)
        {
            IsOnTheFirstLayer = isOnTheFirstLayer;
            if (!isOnTheFirstLayer)
                ActivationFunction = activationFunction;

            IsBias = isBias;
            Synapses = new ObservableCollection<Synapse>();
            Synapses.CollectionChanged += Connections_CollectionChanged;
        }

        public delegate void NeuroSignalEvent(NeuroSignal signal);
        public event NeuroSignalEvent Signal;
        public NeuralNetwork CurrentNeuralNetwork { get; set; }
        public IActivationFunction ActivationFunction { get; set; }
        public bool IsOnTheFirstLayer
        {
            get { return _isOnTheFirstLayer; }
            set
            {
                _isOnTheFirstLayer = value;
                if (value)
                    _stepsToInput = 0;

            }
        }
        public bool IsBias
        {
            get { return _isBias; }
            set
            {
                _isBias = value;
                if (value)
                    _stepsToInput = int.MaxValue;
            }
        }
        public double Sum
        {
            get { return _sum; }
            set { _sum = value; }
        }
        public double OutputValue { get; set; }
        public int StepsToInput { get { return _stepsToInput; } }
        public ObservableCollection<Synapse> Synapses { get; }
        // Dendrites list doesn't include a synapse to bias:
        // Bias does not participate in the forwarding the signals from inputs
        public IEnumerable<Synapse> Dendrites
        {
            get
            {
                if (IsOnTheFirstLayer || IsBias)
                    return new List<Synapse>();

                return Synapses.Where(c =>
                    // PreviousLayer neurons
                    c.GetOtherNeuron(this).StepsToInput == StepsToInput - 1
                    // And not bias
                    && !c.GetOtherNeuron(this).IsBias);
            }
        }
        public IEnumerable<Synapse> Axons
        {
            get
            {
                return Synapses.Where(c =>
                    c.GetOtherNeuron(this).StepsToInput == StepsToInput + 1
                    && !c.GetOtherNeuron(this).IsBias);
            }
        }
        public Synapse SynapseToBias
        {
            get
            {
                return Synapses.FirstOrDefault(s => s.GetOtherNeuron(this).IsBias);
            }
        }
        public bool IsCalculationComplete { get; set; }

        public void Reset()
        {
            _sumLock = false;
            IsCalculationComplete = false;

            if (IsBias)
                OutputValue = BIAS_OUTPUT;
            else
                OutputValue = 0;

            _sum = 0;

            // Bias doesn't send the signal itself, so include it's influence to the sum manually
            if (SynapseToBias != null)
                _sum += SynapseToBias.GetOtherNeuron(this).OutputValue * SynapseToBias.Weight;

            _dendritesInputsConcurrent.Clear();
            _axonsOutputsConcurrent.Clear();
        }

        public void BackPropagate(Synapse connection, double receiversGradient, double step)
        {
            if (IsBias)
                return;

            if (IsOnTheFirstLayer)
                return;

            if (!Axons.Contains(connection))
                return;// And do not throw

            if (ActivationFunction is not ILayerIndependentFunction)
                return;

            _axonsOutputsConcurrent.TryAdd(connection, receiversGradient);
            var backPropagatedConnections = _axonsOutputsConcurrent.Keys;
            // if not all output connections are backPropagated yet, do nothing
            if (!Axons.All(oc => backPropagatedConnections.Contains(oc)))
                return;

            lock (_lockObject)
            {
                if (_backPropagationLock)
                    return;

                _backPropagationLock = true;
            }

            // Otherwise continue back propagation
            // 1. Calculate weighted sum
            var weightedSum = _axonsOutputsConcurrent.Sum(c => c.Value * c.Key.Weight);
            // 2. Calculate local gradient 
            // weightedSum(sigma * omega) * f * (1 - f)
            var activationFunction = ActivationFunction as ILayerIndependentFunction;
            var localGradient = weightedSum * activationFunction.Derivative(OutputValue);
            // 3. Run backPropagations to all other input connections
            //Console.WriteLine($"MidLayer weightedSum = {weightedSum}, local gradient = {localGradient}");
            var synapsesToModify = Dendrites.ToList();
            if (SynapseToBias != null)
                synapsesToModify.Add(SynapseToBias);

            Parallel.ForEach(synapsesToModify, c =>
            {
                var previousLayerNeuron = c.GetOtherNeuron(this);
                c.Weight -= step * localGradient * previousLayerNeuron.OutputValue;
                previousLayerNeuron.BackPropagate(c, localGradient, step);
            });
            _backPropagationLock = false;
        }
        // Recalculate the weight for the way to input for the convergence
        private void Connections_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (IsOnTheFirstLayer)
                return;

            if (IsBias)
                return;

            if (!Synapses.Any())
            {
                _stepsToInput = int.MaxValue;
                //Console.WriteLine($"My layer is {StepsToInput}");
                return;
            }
            _stepsToInput = Synapses.Min(c => c.GetOtherNeuron(this).StepsToInput) + 1;
            //Console.WriteLine($"My layer is {StepsToInput}");
        }
        public IEnumerable<double> FrontPropagate(double inputValue)
        {
            if (!IsOnTheFirstLayer || IsBias)
                throw new InvalidOperationException("This is not an input neuron!");

            OutputValue = inputValue;
            var signal = new NeuroSignal(this, inputValue);
            IEnumerable<double> ret = null;

            Parallel.ForEach(Axons, a =>
            {
                var outputs = a.GetOtherNeuron(this).FrontPropagate(a, signal);
                if (outputs != null && outputs.Any())
                    ret = outputs;
            });
            return ret;
        }
        public IEnumerable<double> FrontPropagate(Synapse synapse, NeuroSignal signal)
        {
            synapse.IsSummarized = true;
            _summarizationBag.Add(signal.Value * synapse.Weight);
            if (!Dendrites.All(d => d.IsSummarized))
                return null;

            lock (_lockObject)
            {
                if (_sumLock)
                    return null;

                _sumLock = true;
            }

            Sum += _summarizationBag.Sum();
            _summarizationBag.Clear();
            Dendrites.ToList().ForEach(d => d.IsSummarized = false);

            if (ActivationFunction is SoftmaxFunction softMax)
            {
                softMax.ApplySum();
            }
            else if (ActivationFunction is ILayerIndependentFunction activationFunction)
            {
                OutputValue = activationFunction.Function(Sum);
                IsCalculationComplete = true;
            }

            if (Axons != null && Axons.Any())
            {
                var neuroSignal = new NeuroSignal(this, OutputValue);
                IEnumerable<double> ret = null;
                Parallel.ForEach(Axons, a =>
                {
                    var nextNeuron = a.GetOtherNeuron(this);
                    var outputs = nextNeuron.FrontPropagate(a, neuroSignal);
                    if (outputs != null && outputs.Any())
                        ret = outputs;
                });
                return ret;
            }
            else if (CurrentNeuralNetwork.OutputNeurons.Contains(this)
                && CurrentNeuralNetwork.OutputNeurons.All(n => n.IsCalculationComplete))
            {
                return CurrentNeuralNetwork.OutputNeurons.Select(n => n.OutputValue);
            }
            return null;
        }
    }

    public class NeuroSignal : EventArgs
    {
        public double Value { get; }
        public Neuron Sender { get; }
        public NeuroSignal(Neuron sender, double value)
        {
            Sender = sender;
            Value = value;
        }
    }
}
