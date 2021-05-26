using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0
{
    public class Sample
    {
        public Sample(IEnumerable<double> inputVector, IEnumerable<double> aeaitedResponse)
        {
            InputVector = inputVector;
            ExpectedResponse = aeaitedResponse;
        }
        public IEnumerable<double> InputVector { get; }
        public IEnumerable<double> ExpectedResponse { get; }
    }

    public class Selection
    {
        private List<Sample> _samples = new List<Sample>();
        private Random _rnd = new Random();
        private List<Sample> _givenSamples = new List<Sample>();
        private List<long> _msPerStep = new List<long>();
        private long _lastStopwatchValue;
        private Stopwatch _stopWatch = new Stopwatch();
        public List<Sample> Samples 
        {
            get { return _samples; } 
            set { _samples = value; }
        }
        public bool HasNextRandomSample { get { return _givenSamples.Count < Samples.Count; } }
        public void ResetRandomizer()
        {
            _givenSamples.Clear();
            _msPerStep.Clear();
            _rnd = new Random(DateTime.Now.Ticks.GetHashCode());
        }
        public Sample GetNextRandomSample()
        {
            if (!_stopWatch.IsRunning)
                _stopWatch.Start();
            else
            {
                var elapsedMs = _stopWatch.ElapsedMilliseconds;
                _msPerStep.Add(elapsedMs - _lastStopwatchValue);
                _lastStopwatchValue = elapsedMs;
            }
            var exceptedSamples = Samples.Except(_givenSamples);
            var randomSample = exceptedSamples.ElementAt(_rnd.Next(0, exceptedSamples.Count()));
            _givenSamples.Add(randomSample);
            double avgSecondsLeft = 0;
            if (_msPerStep.Any()) 
            {
                var averageTimePerStep = _msPerStep.Sum() / _msPerStep.Count;
                avgSecondsLeft = (_samples.Count() - _givenSamples.Count()) * averageTimePerStep / 1000;
            }
            Console.WriteLine($"Samples left: {_samples.Count() - _givenSamples.Count()}, average time left: {avgSecondsLeft}s");
            return randomSample;
        }
    }
}
