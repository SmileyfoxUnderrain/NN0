using System;
using System.Collections.Generic;
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
        public List<Sample> Samples 
        {
            get { return _samples; } 
            set { _samples = value; }
        }
        public bool HasNextRandomSample { get { return _givenSamples.Count < Samples.Count; } }
        public void ResetRandomizer()
        {
            _givenSamples.Clear();
            _rnd = new Random(DateTime.Now.Ticks.GetHashCode());
        }
        public Sample GetNextRandomSample()
        {
            var exceptedSamples = Samples.Except(_givenSamples);
            var randomSample = exceptedSamples.ElementAt(_rnd.Next(0, exceptedSamples.Count()));
            _givenSamples.Add(randomSample);
            return randomSample;
        }
    }
}
