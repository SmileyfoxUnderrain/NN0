using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.LossFunctions
{
    public class MeanAbsolutePercentageErrorFunction : ILossFunction
    {
        public double CalculateLoss(IEnumerable<Tuple<double, double>> receivedToExpectedOutputs)
        {
            // Division by zero avoidance
            var nonZeroOutputs = receivedToExpectedOutputs.Where(t => t.Item2 != 0);
            var sum = nonZeroOutputs.Sum(t => 100 * Math.Abs(t.Item1 - t.Item2) / t.Item2);
            var ret = sum / receivedToExpectedOutputs.Count();
            return ret;
        }
    }
}
