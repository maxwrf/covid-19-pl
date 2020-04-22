## Simple Sir App
This app is able to generate predictions about the evolution of a pandemic based on user input and by leveraging the SIR model. Additionally, the app offers an introduction to the maths of SIR model. Finally, the SIR model is also fitted to historic data from the Covid-19 outbreak in Portugal (the Beta and Gamma parameters are estimated using scipy by minimizing error the error between predicted and actual infections). For the forward integration in the SIR model two solvers have been implemented in plain Python, namely the Euler Forward and the Runge–Kutta4 methods. 

### References
- https://towardsdatascience.com/infectious-disease-modelling-beyond-the-basic-sir-model-216369c584c4
- https://www.kaggle.com/anjum48/seir-hcd-model
- https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions
- https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
- https://www.youtube.com/watch?v=mwJXjxMTwAw&t=1129s
- https://en.wikipedia.org/wiki/Euler_method
- https://gitlab.com/dice89/python-plotting-api
