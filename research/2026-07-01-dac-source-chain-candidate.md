# DAC/Modulator Source-Chain Candidate Search

## Question

Find a single DAC/modulator source-chain candidate that simultaneously supports:

- `10 GS/s` or faster input update rate.
- `>= 7.35 ENOB` or equivalent usable precision.
- `>= 50 dBc SFDR`.
- Near `2.5 pJ/sample` DAC-side energy.

## Result

No single source-chain candidate found yet.

The strongest concrete candidate is the ADI AD9172 family as a commercial
direct-RF DAC reference. It clears the update-rate and SFDR side of the
requirement, but not the energy target: the public ADI product brief lists
`1.5 W/channel at 12 GSPS`, which is about `125 pJ/sample/channel`.
That is roughly `50x` the `2.5 pJ/sample` target before adding a modulator
driver or optical modulator. The data sheet also gives static INL/DNL, but does
not provide the DAC ENOB target directly.

## Candidate Check

| Candidate | Rate | Precision / Linearity | Energy | Verdict |
| --- | ---: | --- | ---: | --- |
| ADI AD9172 dual 16-bit RF DAC | 12-12.6 GSPS | Product brief: 70 dBc SFDR at 1.8 GHz, 12 GSPS; data sheet: 16-bit nominal, +/-7 LSB INL/DNL | about 125 pJ/sample/channel from 1.5 W/channel at 12 GSPS | Best rate/SFDR reference, but energy fails by about 50x and ENOB is not stated |
| 10 GS/s 8-bit current-steering DAC in 65 nm CMOS | 10 GS/s | Simulated SFDR up to 43 dB over Nyquist | 37.5 mW, about 3.75 pJ/sample | Close on energy and rate, but SFDR fails the 50 dBc target and result is simulated |
| 100 GS/s 8-bit distributed DAC in 28 nm CMOS | 100 GS/s | ENOB/SFDR range 5.3 bit/41 dB down to 3.2 bit/27 dB | 2.5 W, about 25 pJ/sample at 100 GS/s | Rate clears easily, but precision, SFDR, and energy miss |
| 10-bit DC-20 GHz MRZ DAC | 3.35 GS/s DAC with RF synthesis to 20 GHz | Greater than 48 dB SFDR from DC to 20 GHz | 1.91 W, about 570 pJ/sample at 3.35 GS/s | Strong RF linearity idea, but sample rate/energy miss and SFDR is still below 50 dBc |

## Interpretation

The search makes the DAC hole sharper rather than closing it. The published
device space splits into two camps:

- Low-energy/high-rate academic CMOS DACs can approach the energy target, but
  the available examples do not meet the `>= 50 dBc` linearity requirement.
- Commercial RF DACs meet rate and SFDR, but their channel power is far above
  the optical-inference energy budget.

The current `2.5 pJ/sample` DAC target should therefore remain a requirement
ceiling, not a demonstrated component. A practical next step is to decide whether
the architecture should relax DAC precision, reduce electrical DAC count even
further, use lower-resolution optical coding plus calibration, or treat the DAC
as a research blocker for the cheap-inference path.

## Sources

- Analog Devices, "AD9172: Dual, 16-Bit, 12.6 GSPS RF DAC with Channelizers."
  URL: https://www.analog.com/media/en/technical-documentation/data-sheets/ad9172.pdf
- Analog Devices, "High Speed Converters Lead Industry with 28 nm CMOS
  Technology."
  URL: https://www.analog.com/media/en/news-marketing-collateral/product-highlight/ad9208-ad9172-high-speed-converters.pdf
- WCSE, "A 10GS/s 8-bit Current Steering DAC in 65nm CMOS Technology."
  URL: https://www.wcse.org/content-14-499-1.html
- University of Stuttgart, "Digital-to-Analog-Converter with 100 GS/s."
  URL: https://www.int.uni-stuttgart.de/en/research/ic/dac100g/
- Lucas Duncan, "A 10-bit DC-20 GHz Multiple-Return-To-Zero DAC with >48 dB
  SFDR."
  URL: https://rave.ohiolink.edu/etdc/view?acc_num=osu1492740839889776
