from __future__ import annotations

from pathlib import Path

import yaml

from automation.k8s_autoscaler import (
    AutoScalingConfig,
    ClusterAutoscalerConfig,
    HorizontalPodAutoscalerConfig,
    generate_manifests,
    write_manifests,
)


def test_generate_manifests_structure(tmp_path: Path) -> None:
    config = AutoScalingConfig(
        cluster_name="tessrax-test",
        hpa=HorizontalPodAutoscalerConfig(
            name="truth-api-hpa",
            namespace="tessrax",
            deployment="truth-api",
            min_replicas=2,
            max_replicas=5,
            target_cpu_percentage=60,
        ),
        cluster_autoscaler=ClusterAutoscalerConfig(
            namespace="kube-system",
            service_account="cluster-autoscaler",
            min_nodes=1,
            max_nodes=4,
            scale_up_threshold=0.7,
            scale_down_threshold=0.4,
        ),
    )

    manifests = generate_manifests(config)

    hpa = manifests["horizontal_pod_autoscaler"]
    assert hpa["kind"] == "HorizontalPodAutoscaler"
    assert hpa["spec"]["scaleTargetRef"]["name"] == "truth-api"
    assert hpa["spec"]["maxReplicas"] == 5

    cluster = manifests["cluster_autoscaler"]
    deployment = next(item for item in cluster if item["kind"] == "Deployment")
    assert deployment["metadata"]["labels"]["app"] == "cluster-autoscaler"
    command = deployment["spec"]["template"]["spec"]["containers"][0]["command"]
    assert any(flag.startswith("--scale-down-utilization-threshold=") for flag in command)

    written = write_manifests(manifests, tmp_path)
    assert all(path.exists() for path in written)
    round_trip = yaml.safe_load((tmp_path / "horizontal_pod_autoscaler.yaml").read_text())
    assert round_trip["metadata"]["name"] == "truth-api-hpa"
