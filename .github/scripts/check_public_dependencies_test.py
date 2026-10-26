import pytest
from check_public_dependencies import violations


def package():
    return {
        "name": "example",
        "source": {"registry": "https://pypi.org/simple"},
        "sdist": {"url": "https://files.pythonhosted.org/packages/example.tar.gz"},
        "wheels": [{"url": "https://files.pythonhosted.org/packages/example.whl"}],
    }


def test_accepts_public_distributions_and_local_project():
    assert (
        violations(
            {"package": [package(), {"name": "stitch", "source": {"editable": "."}}]}
        )
        == []
    )


@pytest.mark.parametrize(
    "source",
    [
        {
            "registry": "https://private.d.codeartifact.us-east-1.amazonaws.com/pypi/simple/"
        },
        {"registry": "https://user:password@pypi.org/simple"},
        {"registry": "https://pypi.org.example.com/simple"},
        {"url": "https://private.example.com/example.whl"},
    ],
)
def test_rejects_non_public_sources(source):
    dependency = package()
    dependency["source"] = source
    assert violations({"package": [dependency]})


@pytest.mark.parametrize("artifact", ["sdist", "wheels"])
@pytest.mark.parametrize(
    "url",
    [
        "https://private.d.codeartifact.us-east-1.amazonaws.com/example.whl",
        "https://user:password@files.pythonhosted.org/example.whl",
        "https://files.pythonhosted.org.example.com/example.whl",
        "http://files.pythonhosted.org/example.whl",
    ],
)
def test_rejects_non_public_distribution_urls(artifact, url):
    dependency = package()
    dependency[artifact] = [{"url": url}] if artifact == "wheels" else {"url": url}
    assert violations({"package": [dependency]})
