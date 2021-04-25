using System;
using System.Collections.Specialized;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace NN0
{
    public class Neuron
    {
        private Dictionary<Synapse, double> _dendritesInputValues = new Dictionary<Synapse, double>();
        private Dictionary<Synapse, double> _outputSynapsesGradients = new Dictionary<Synapse, double>();
        private bool _isOnTheFirstLayer;
        private bool _isBias;
        private double _sum;
        protected int _stepsToInput = int.MaxValue;

        // Neuron needs to be synchronized in the network
        public Neuron(bool isOnTheFirstLayer = false, bool isBias = false)
        {
            IsOnTheFirstLayer = isOnTheFirstLayer;
            IsBias = isBias;
            Synapses = new ObservableCollection<Synapse>();
            Synapses.CollectionChanged += Connections_CollectionChanged;
        }

        public delegate void NeuroSignalEvent(NeuroSignal signal);
        public event NeuroSignalEvent Signal;
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
            set 
            { 
                _sum = value;
                OutputValue = ActivationFunction(_sum);
            }
        }
        public double OutputValue { get; set; }
        public int StepsToInput { get { return _stepsToInput; } }
        public ObservableCollection<Synapse> Synapses { get; }
        public IEnumerable<Synapse> Dendrites
        {
            get
            {
                if (IsOnTheFirstLayer || IsBias)
                    return new List<Synapse>();

                return Synapses.Where(c => 
                    // PreviousLayer neurons
                    c.GetOtherNeuron(this).StepsToInput == StepsToInput - 1
                    // Or Bias
                    || c.GetOtherNeuron(this).IsBias);
            }
        }
        public IEnumerable<Synapse> Axons
        {
            get
            {
                return Synapses.Where(c => c.GetOtherNeuron(this).StepsToInput == StepsToInput + 1);
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
        public void OnIncomingSignal(NeuroSignal signal)
        {
            //Console.WriteLine($"Layer {StepsToInput}. Got signal {signal.Value}");
            if (signal == null)
                return;

            // Do not handle the signals from the current layer or higher layers
            var synapse = Dendrites.Where(c => c.GetOtherNeuron(this) == signal.Sender).FirstOrDefault();
            if (synapse == null)
                return;

            _dendritesInputValues.Add(synapse, signal.Value);

            // If not all dendrites sent signal yet, do nothing
            if (!Dendrites.All(d => _dendritesInputValues.Keys.Contains(d)))
                return;

            // Else calculate sum and output value and send to the axons
            Sum = _dendritesInputValues.Sum(kvp => kvp.Key.Weight * kvp.Value);
            IsCalculationComplete = true;
            // If no next layer neurons subscribed for Signal event, do nothing
            if (Signal == null)
                return;

            var neuroSignal = new NeuroSignal(this, OutputValue);
            Signal(neuroSignal);
        }
        /// <summary>
        /// Receive signal to the input neuron and send it to others
        /// Input neuron does not handles the signal with activation function
        /// </summary>
        /// <param name="value">Input signal value</param>
        public void ReceiveSignal(double value)
        {
            if (!IsOnTheFirstLayer)
                return;

            OutputValue = value;
            Signal(new NeuroSignal(this, OutputValue));
        }

        public void Reset() 
        {
            IsCalculationComplete = false;
            OutputValue = 0;
            Sum = 0;
            // Bias doesn't send the signal itself, so include it to the sum manually
            if (SynapseToBias != null)
                Sum += SynapseToBias.GetOtherNeuron(this).OutputValue * SynapseToBias.Weight;

            _dendritesInputValues.Clear();
            _outputSynapsesGradients.Clear();
        }

        public void BackPropagate(Synapse connection, double receiversGradient, double step)
        {
            if (IsOnTheFirstLayer)
                return;

            if (!Axons.Contains(connection))
                return;// And do not throw

            _outputSynapsesGradients.Add(connection, receiversGradient);
            var backPropagatedConnections = _outputSynapsesGradients.Keys;
            // if not all output connections are backPropagated yet, do nothing
            if (!Axons.All(oc => backPropagatedConnections.Contains(oc)))
                return;

            // Otherwise continue back propagation
            // 1. Calculate weighted sum
            var weightedSum = _outputSynapsesGradients.Sum(c => c.Value * c.Key.Weight);
            // 2. Calculate local gradient 
            // weightedSum(sigma * omega) * f * (1 - f)
            var localGradient = weightedSum * ActivationFunctionDerivative(OutputValue);
            // 3. Run backPropagations to all other input connections
            //Console.WriteLine($"MidLayer weightedSum = {weightedSum}, local gradient = {localGradient}");
            this.Dendrites.ToList().ForEach(c =>
            {
                var previousLayerNeuron = c.GetOtherNeuron(this);
                c.Weight -= step * localGradient * previousLayerNeuron.OutputValue;
                previousLayerNeuron.BackPropagate(c, localGradient, step);
            });

        }
        // Recalculate the weight for the way to input for the convergence
        private void Connections_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (IsOnTheFirstLayer)
                return;

            if (!Synapses.Any())
            {
                _stepsToInput = int.MaxValue;
                Console.WriteLine($"My layer is {StepsToInput}");
                return;
            }

            _stepsToInput = Synapses.Min(c => c.GetOtherNeuron(this).StepsToInput) + 1;
            Console.WriteLine($"My layer is {StepsToInput}");
        }
        // TODO make activationFunction changeable
        private double ActivationFunction(double x)
        {
            var ret = 1 / (1 + Math.Pow(Math.E, x * (-1)));
            return ret;
        }
        private double ActivationFunctionDerivative(double x)
        {
            var ret = x * (1 - x);
            return ret;
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
