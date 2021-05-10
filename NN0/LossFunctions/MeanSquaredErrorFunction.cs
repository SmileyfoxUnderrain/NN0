using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.LossFunctions
{
    public class MeanSquaredErrorFunction : ILossFunction
    {
        public double CalculateLoss(IEnumerable<Tuple<double, double>> receivedToExpectedOutputs)
        {
            var sum = receivedToExpectedOutputs
               .Sum(t => Math.Pow(t.Item1 - t.Item2, 2));
            var ret = sum / receivedToExpectedOutputs.Count();
            return ret;
        }
    }
}
