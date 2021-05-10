using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public class SoftSignFunction : IActivationFunction
    {
        public double Function(double x)
        {
            var ret = x / (1 + Math.Abs(x));
            return ret;
        }
        public double Derivative(double x)
        {
            var ret = 1 / Math.Pow(1 + Math.Abs(x), 2);
            return ret;
        }
    }
}
