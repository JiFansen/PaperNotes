#https://statswithr.github.io/book/inference-and-decision-making-with-multiple-parameters.html#sec:NG-MC
library(statsr)
library(ggplot2)
library(PairedData)
library(dplyr)
library(ggthemes)
data("tapwater")
glimpse(tapwater)
#prior hyperparameters.
m_0 <- 35; n_0 <- 25; s2_0 <- 156.25; v_0 <- n_0 - 1
#sample summaries.
Y <- tapwater$tthm
ybar <- mean(Y)
s2 <- var(Y)
n <- length(Y)
#posterior hyperparameters
n_n <- n_0 + n
m_n <- (n*ybar + n_0*m_0)/n_n
v_n <- v_0 + n
s2_n <- ((n-1)*s2 + v_0*s2_0 + n_0*n*(m_0 - ybar)^2/n_n)/v_n
m_n + qt(c(0.025, 0.975), v_n)*sqrt(s2_n/n_n)
bayes_inference(tthm, data=tapwater,
                prior="NG",
                mu_0 = m_0, n_0=n_0, 
                s_0 = sqrt(s2_0), v_0 = v_0,
                stat="mean", type="ci",
                method="theoretical", 
                show_res=T, 
                show_summ=T,
                show_plot=T)
ggplot(data=tapwater, aes(x=tthm)) + geom_histogram()
set.seed(42)
phi <- rgamma(1000, shape = v_n/2, rate = s2_n*v_n/2)
df <- data.frame(phi = sort(phi))
df <- mutate(df, density = dgamma(phi, shape = v_n/2, rate = s2_n*v_n/2))
ggplot(data=df, aes(x=phi)) + 
  geom_histogram(aes(x=phi, y=..density..), bins = 50)+
  geom_density(aes(phi, ..density..), color="black")+ 
  geom_line(aes(x=phi, y=density), color="orange") +
  xlab(expression(phi)) + theme_tufte()
mean(phi)
quantile(phi, c(0.025, 0.975))
1/s2_n
qgamma(c(0.025, 0.975), shape = v_n/2, rate = s2_n*v_n/2)
sigma <- 1/sqrt(phi)
mean(sigma)
quantile(sigma, c(0.025, 0.975))
################################### Chapter 6: Introduction to Bayesian Regression. ###########
library(BAS)
data(bodyfat)
summary(bodyfat)
# Frequentist OLS linear regression
bodyfat.lm = lm(Bodyfat ~ Abdomen, data = bodyfat)
summary(bodyfat.lm)
# Extract coefficients
beta = coef(bodyfat.lm)

# Visualize regression line on the scatter plot
library(ggplot2)
ggplot(data = bodyfat, aes(x = Abdomen, y = Bodyfat)) +
  geom_point(color = "blue") +
  geom_abline(intercept = beta[1], slope = beta[2], size = 1) +
  xlab("abdomen circumference (cm)") 
# Obtain residuals and n
resid = residuals(bodyfat.lm)
n = length(resid)

# Calculate MSE
MSE = 1/ (n - 2) * sum((resid ^ 2))
MSE
# Combine residuals and fitted values into a data frame
result = data.frame(fitted_values = fitted.values(bodyfat.lm),
                    residuals = residuals(bodyfat.lm))

# Load library and plot residuals versus fitted values
library(ggplot2)
ggplot(data = result, aes(x = fitted_values, y = residuals)) +
  geom_point(pch = 1, size = 2) + 
  geom_abline(intercept = 0, slope = 0) + 
  xlab(expression(paste("fitted value ", widehat(Bodyfat)))) + 
  ylab("residuals")
# Find the observation with the largest fitted value
which.max(as.vector(fitted.values(bodyfat.lm)))
# Shows this observation has the largest Abdomen
which.max(bodyfat$Abdomen)
plot(bodyfat.lm, which = 2)
library(ggplot2)       
# Construct current prediction
alpha = bodyfat.lm$coefficients[1]
beta = bodyfat.lm$coefficients[2]
new_x = seq(min(bodyfat$Abdomen), max(bodyfat$Abdomen), 
            length.out = 100)
y_hat = alpha + beta * new_x

# Get lower and upper bounds for mean
ymean = data.frame(predict(bodyfat.lm,
                           newdata = data.frame(Abdomen = new_x),
                           interval = "confidence",
                           level = 0.95))

# Get lower and upper bounds for prediction
ypred = data.frame(predict(bodyfat.lm,
                           newdata = data.frame(Abdomen = new_x),
                           interval = "prediction",
                           level = 0.95))

output = data.frame(x = new_x, y_hat = y_hat, ymean_lwr = ymean$lwr, ymean_upr = ymean$upr, 
                    ypred_lwr = ypred$lwr, ypred_upr = ypred$upr)

# Extract potential outlier data point
outlier = data.frame(x = bodyfat$Abdomen[39], y = bodyfat$Bodyfat[39])

# Scatter plot of original
plot1 = ggplot(data = bodyfat, aes(x = Abdomen, y = Bodyfat)) + geom_point(color = "blue")

# Add bounds of mean and prediction
plot2 = plot1 + 
  geom_line(data = output, aes(x = new_x, y = y_hat, color = "first"), lty = 1) +
  geom_line(data = output, aes(x = new_x, y = ymean_lwr, lty = "second")) +
  geom_line(data = output, aes(x = new_x, y = ymean_upr, lty = "second")) +
  geom_line(data = output, aes(x = new_x, y = ypred_upr, lty = "third")) +
  geom_line(data = output, aes(x = new_x, y = ypred_lwr, lty = "third")) + 
  scale_colour_manual(values = c("orange"), labels = "Posterior mean", name = "") + 
  scale_linetype_manual(values = c(2, 3), labels = c("95% CI for mean", "95% CI for predictions")
                        , name = "") + 
  theme_bw() + 
  theme(legend.position = c(1, 0), legend.justification = c(1.5, 0))

# Identify potential outlier
plot2 + geom_point(data = outlier, aes(x = x, y = y), color = "orange", pch = 1, cex = 6)
pred.39 = predict(bodyfat.lm, newdata = bodyfat[39, ], interval = "prediction", level = 0.95)
out = cbind(bodyfat[39,]$Abdomen, pred.39)
colnames(out) = c("abdomen", "prediction", "lower", "upper")
out





