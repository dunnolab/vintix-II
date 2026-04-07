from dataclasses import dataclass


###############  All available environemnts ###############
#
#     "citylearn_challenge_2023_phase_3_2",
#     "citylearn_challenge_2023_phase_2_online_evaluation_1",
#     "citylearn_challenge_2020_climate_zone_1",
#     "citylearn_challenge_2022_phase_2",
#     "vt_chittenden_county_neighborhood",
#     "citylearn_challenge_2022_phase_1",
#     "citylearn_challenge_2022_phase_all",
#     "baeda_3dem",
#     "citylearn_challenge_2020_climate_zone_2",
#     "ca_alameda_county_neighborhood",
#     "citylearn_challenge_2020_climate_zone_4",
#     "citylearn_challenge_2021",
#     "tx_travis_county_neighborhood",
#     "citylearn_challenge_2023_phase_2_local_evaluation",
#     "citylearn_challenge_2023_phase_2_online_evaluation_2",
#     "citylearn_challenge_2020_climate_zone_3",
#     "citylearn_challenge_2023_phase_3_1",
#     "citylearn_challenge_2023_phase_2_online_evaluation_3",
#     "citylearn_challenge_2023_phase_3_3",
#     "citylearn_challenge_2022_phase_3",
#     "citylearn_challenge_2023_phase_1"
#
###########################################################

ALIAS_TO_ENV = {
    "2022_phase_1": "citylearn_challenge_2022_phase_1",
    "2022_phase_2": "citylearn_challenge_2022_phase_2",
}

@dataclass
class TaskInfo:
    env_name: str
    start_time_step: int
    end_time_step: int

TRAIN_TASKS = {
    **{f"2022_phase_1_task_{idx:02d}": TaskInfo(ALIAS_TO_ENV["2022_phase_1"], idx * 730, idx * 730 + 729) for idx in range(0, 10)},
    **{f"2022_phase_2_task_{idx:02d}": TaskInfo(ALIAS_TO_ENV["2022_phase_2"], idx * 730, idx * 730 + 729) for idx in range(0, 10)}
}

TEST_TASKS = {
    **{f"2022_phase_1_task_{idx:02d}": TaskInfo(ALIAS_TO_ENV["2022_phase_1"], idx * 730, idx * 730 + 729) for idx in range(10, 12)},
    **{f"2022_phase_2_task_{idx:02d}": TaskInfo(ALIAS_TO_ENV["2022_phase_2"], idx * 730, idx * 730 + 729) for idx in range(10, 12)}
}

ALL_TASKS = {**TRAIN_TASKS, **TEST_TASKS}
