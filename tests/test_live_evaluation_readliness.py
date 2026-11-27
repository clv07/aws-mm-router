"""Regression coverage for fair live AWS SLO evaluation and CSV append."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import main


def test_live_trace_requires_measured_slos():
    old = [{'id': '1', 'arrival_s': '0.1', 'run': '00',
            'modality': 'chat', 'size': 'short', 'input': '1'}]
    with pytest.raises(ValueError, match='generate a new trace'):
        main.validate_live_trace(old, 'legacy.csv')


def test_live_trace_accepts_valid_request_slos():
    q = {'id': '1', 'arrival_s': '0.1', 'run': '00', 'modality': 'chat',
         'size': 'short', 'input': '1', 'deadline_s': '5',
         'ttft_deadline_s': '2', 'tpot_slo_ms': '100'}
    assert main.validate_live_trace([q], 'trace.csv') == [q]


def test_arrival_relative_slo_and_failed_measurements():
    q = {'deadline_s': '10', 'ttft_deadline_s': '4', 'tpot_slo_ms': '100'}
    success = {'status': 'ok', 'e2e_s': 9.5, 'ttft_s': 3.5, 'tpot_ms': 90}
    main.attach_live_slos(success, q)
    assert all(success[key] is True for key in (
        'completion_slo_met', 'ttft_slo_met', 'tpot_slo_met', 'overall_slo_met'))
    assert success['deadline_s'] == 10

    failed = {'status': 'error', 'e2e_s': 1, 'ttft_s': None, 'tpot_ms': None}
    main.attach_live_slos(failed, q)
    assert failed['overall_slo_met'] is False
    assert failed['completion_slo_met'] is False

    partial = {'status': 'ok', 'e2e_s': 8, 'ttft_s': None, 'tpot_ms': 80}
    main.attach_live_slos(partial, q)
    assert partial['completion_slo_met'] is True
    assert partial['overall_slo_met'] is False


def test_existing_csv_schema_requires_manual_cleanup(tmp_path):
    request = tmp_path / 'requests.csv'
    request.write_text('id,old_column\n1,hello\n')
    with pytest.raises(RuntimeError, match='different column format'):
        main.append_rows(request, main.REQUEST_FIELDS, [{'id': '2'}])
    assert request.read_text() == 'id,old_column\n1,hello\n'


def test_current_csv_schema_appends_across_policies(tmp_path):
    request = tmp_path / 'requests.csv'
    main.append_rows(request, main.REQUEST_FIELDS, [{'policy': 'rr', 'id': '1'}])
    main.append_rows(request, main.REQUEST_FIELDS, [{'policy': 'ect', 'id': '2'}])
    main.append_rows(request, main.REQUEST_FIELDS, [{'policy': 'adaptive', 'id': '3'}])
    import csv
    with request.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [row['policy'] for row in rows] == ['rr', 'ect', 'adaptive']


def test_live_evaluation_summary_counts_failures(monkeypatch):
    captured = []
    monkeypatch.setattr(main, 'append_rows', lambda path, fields, rows: captured.extend(rows))
    rows = [
        dict(modality='chat', node='amd', status='ok', e2e_s=5., ttft_s=2.,
             queue_s=0., completion_slo_met=True, ttft_slo_met=True,
             tpot_slo_met=True, overall_slo_met=True),
        dict(modality='chat', node='amd', status='error', e2e_s=7., ttft_s=None,
             queue_s=0., completion_slo_met=False, ttft_slo_met=False,
             tpot_slo_met=False, overall_slo_met=False),
    ]
    main.evaluate_run(rows, {'amd': {}}, {'run_id': 'test'}, 2., 10., {})
    total = next(r for r in captured if r['group'] == 'all')
    assert total['requests'] == 2
    assert total['success'] == 1
    assert total['overall_slo_attainment'] == .5
    assert total['completion_slo_attainment'] == .5


