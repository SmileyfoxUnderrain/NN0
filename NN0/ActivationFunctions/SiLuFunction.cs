using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public class SiLuFunction : IActivationFunction
    {
        public double Function(double x)
        {
            var ret = x / (1 + Math.Pow(Math.E, -1 * x));
            return ret;
        }
        public double Derivative(double x)
        {
            var e = Math.E;
            var mx = -1 * x;

            var ret = (1 + Math.Pow(e, mx) + x * Math.Pow(e, mx)) / Math.Pow(1 + Math.Pow(e, mx), 2);
            return ret;
        }
    }
}
