from types import SimpleNamespace

from src import tune_satisfaction_lightgbm as tune_sat_lgb


def test_parse_seed_text_uses_default_and_parses_values():
    assert tune_sat_lgb.parse_seed_text(None, '42,52') == [42, 52]
    assert tune_sat_lgb.parse_seed_text('7, 8,9', '1') == [7, 8, 9]


def test_objective_from_summary_supports_all_modes():
    summary = {
        'holdout_qwk_mean': 0.11,
        'holdout_macro_f1_mean': 0.22,
        'holdout_weighted_f1_mean': 0.33
    }

    assert tune_sat_lgb.objective_from_summary(summary, 'holdout', 'qwk') == 0.11
    assert tune_sat_lgb.objective_from_summary(summary, 'holdout', 'macro_f1') == 0.22
    assert tune_sat_lgb.objective_from_summary(summary, 'holdout', 'weighted_f1') == 0.33
    assert tune_sat_lgb.objective_from_summary(summary, 'holdout', 'qwk_macro_blend') == 0.1375


def test_build_output_paths_creates_study_directory():
    paths = tune_sat_lgb.build_output_paths('sat_test_study', tune_sat_lgb.OUTPUT_ROOT)

    assert paths['study_dir'] == tune_sat_lgb.OUTPUT_ROOT / 'sat_test_study'
    assert paths['study_dir'].exists()
    assert paths['storage_path'].name == 'sat_test_study.sqlite3'
    assert paths['trials_path'].parent == paths['study_dir']


def test_resolve_storage_defaults_to_in_memory():
    args = SimpleNamespace(storage_path=None, persist_study=False, resume=False)
    output_paths = tune_sat_lgb.build_output_paths('sat_storage_default', tune_sat_lgb.OUTPUT_ROOT)

    storage_cfg = tune_sat_lgb.resolve_storage(args, output_paths)

    assert storage_cfg['persist_study'] is False
    assert storage_cfg['storage_path'] is None
    assert storage_cfg['storage_uri'] is None


def test_resolve_storage_requires_persistence_for_resume():
    args = SimpleNamespace(storage_path=None, persist_study=False, resume=True)
    output_paths = tune_sat_lgb.build_output_paths('sat_storage_resume', tune_sat_lgb.OUTPUT_ROOT)

    try:
        tune_sat_lgb.resolve_storage(args, output_paths)
    except ValueError as exc:
        assert '--resume requires --persist-study' in str(exc)
    else:
        raise AssertionError('Expected resolve_storage to reject resume without persistence')


def test_default_anchor_params_match_tunable_keys():
    anchor = tune_sat_lgb.default_anchor_params()

    assert set(anchor) == set(tune_sat_lgb.tune_param_keys())
    assert anchor['n_estimators'] > 0
    assert anchor['num_leaves'] > 0
    assert anchor['reg_alpha'] > 0
    assert anchor['reg_lambda'] > 0
    assert anchor['max_bin'] in {63, 127, 255}
