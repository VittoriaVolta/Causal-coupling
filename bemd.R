library(Rlibeemd)
library(tidyverse)
library(tidyquant)
library(readxl)
library(openxlsx)

signal <- vol_data[, 1][["EST"]] + vol_data[, 2][["UKX"]] * 1i
imfs <- bemd(signal)

# plot the data and the imfs
oldpar <- par()
par(mfrow = c(5, 1), mar = c(0.5, 4.5, 0.5, 0.5), oma = c(4, 0, 2, 0))
ts.plot(EST_UKX_no_accounc_1145, col = 1:2, lty = 1:2, ylab = "signal", gpars = list(xaxt = "n"))
for(i in 1:9) {
  ts.plot(Re(imfs[, i]), Im(imfs[, i]), col = 1:2, lty = 1:2,
          ylab = if(i < 4) paste("IMF", i) else "residual", gpars = list(xaxt = "n"))
}
axis(1)
title(xlab = "Time (days)", main = "Bivariate EMD decomposition", outer = TRUE)
par(oldpar)

write.xlsx(imfs, "bemd_imfs.xlsx")
