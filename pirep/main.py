from defs.report import PilotReport

rep = PilotReport.parse(
    "UA /OV CYZP 180045 /TM 0015 /FLDURC /TP DH8D /IC MOD MIX 100-150 LGTMIX 150-200"
)
print(rep)
