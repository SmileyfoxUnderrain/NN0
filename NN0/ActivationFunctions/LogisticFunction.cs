using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class LogisticFunction : ILayerIndependentFunction
    {
        public double Function(double x)
        {
            var ret = 1 / (1 + Math.Pow(Math.E, -1 * x));
            return ret;
        }
        public double Derivative(double x)
        {
            var ret = x * (1 - x);
            return ret;
        }

    }
}
