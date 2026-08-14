from fastwam.policy_environment import standalone_policy_environment


def test_standalone_policy_environment_removes_only_exact_simulator_path():
    source = {
        "SIM_SITE_PACKAGES": "/tmp/libero-site",
        "PYTHONPATH": "/project:/tmp/libero-site-extra:/tmp/libero-site:/diffusers",
        "KEEP": "value",
    }

    result = standalone_policy_environment(source)

    assert result["PYTHONPATH"] == "/project:/tmp/libero-site-extra:/diffusers"
    assert result["KEEP"] == "value"
    assert source["PYTHONPATH"].endswith("/tmp/libero-site:/diffusers")


def test_standalone_policy_environment_preserves_order_and_duplicates():
    result = standalone_policy_environment(
        {
            "SIM_SITE_PACKAGES": "/sim",
            "PYTHONPATH": "/a:/b:/sim:/a:/c",
        }
    )

    assert result["PYTHONPATH"] == "/a:/b:/a:/c"


def test_standalone_policy_environment_is_unchanged_without_contract_variables():
    assert standalone_policy_environment({"PYTHONPATH": "/a:/b"}) == {
        "PYTHONPATH": "/a:/b"
    }
    assert standalone_policy_environment({"SIM_SITE_PACKAGES": "/sim"}) == {
        "SIM_SITE_PACKAGES": "/sim"
    }
