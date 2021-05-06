using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public class FunctionFactory
    {
        public static IActivationFunction GetByType(ActivationFunctionType type)
        {
            if (type == ActivationFunctionType.Arctg)
                return new ArctgFunction();
            if (type == ActivationFunctionType.BentIdentity)
                return new BentIdentityFunction();
            if (type == ActivationFunctionType.BinaryStep)
                return new BinaryStepFunction();
            if (type == ActivationFunctionType.Gaussian)
                return new GaussianFunction();
            if (type == ActivationFunctionType.Identiy)
                return new IdentityFunction();
            if (type == ActivationFunctionType.LeakyReLU)
                return new LeakyReLuFunction();
            if (type == ActivationFunctionType.Logistic)
                return new LogisticFunction();
            if (type == ActivationFunctionType.ReLU)
                return new ReLuFunction();
            if (type == ActivationFunctionType.SiLU)
                return new SiLuFunction();
            if (type == ActivationFunctionType.Sinc)
                return new SincFunction();
            if (type == ActivationFunctionType.Softplus)
                return new SoftplusFunction();
            if (type == ActivationFunctionType.Softsign)
                return new SoftSignFunction();
            if (type == ActivationFunctionType.Tanh)
                return new TahnFunction();
             
            throw new NotImplementedException("Unknown ActivationFunction Type");
        }
    }
}
