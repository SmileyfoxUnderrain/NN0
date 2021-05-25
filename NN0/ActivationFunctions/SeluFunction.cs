using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class SeluFunction : ILayerIndependentFunction
    {
        private const double LAMBDA = 1.0507;
        private const double ALPHA = 1.67326;
        public double Function(double x)
        {
            var ret = x >= 0
                ? LAMBDA * x
                : LAMBDA * ALPHA * (Math.Pow(Math.E, x) - 1);

            return ret;
        }
        public double Derivative(double x)
        {
            var ret = x >= 0
                ? LAMBDA
                : LAMBDA * ALPHA * Math.Pow(Math.E, x);

            return ret;
        }
    }
}
