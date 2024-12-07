from defs.report import PilotReport

rep = PilotReport.parse(
    "CNY UA /OV OAB055019/TM 0034/FL200/TP E2/TB OCNL LGT CHOP/RM MOSTLY SMOOTH ZDVWC-"
)
print(rep)
