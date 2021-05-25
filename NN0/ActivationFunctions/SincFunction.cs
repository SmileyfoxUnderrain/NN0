using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class SincFunction : ILayerIndependentFunction
    {
        public double Function(double x)
        {
            if (x == 0)
                return 1;

            var ret = Math.Sin(x) / x;
            return ret;
        }
        public double Derivative(double x)
        {
            if (x == 0)
                return 0;

            var ret = (Math.Cos(x) / x) - (Math.Sin(x) / Math.Pow(x, 2));
            return ret;
        }
    }
}
