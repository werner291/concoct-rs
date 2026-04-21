{ buildPythonPackage, fetchPypi, setuptools, biopython, six }:

buildPythonPackage {
  pname = "bcbio-gff";
  version = "0.7.1";
  pyproject = true;

  src = fetchPypi {
    pname = "bcbio-gff";
    version = "0.7.1";
    hash = "sha256-0dwylBR7lbrO1gM/Y4ag/tRcQ3Z+8C0SI99e9JfpzKY=";
  };

  build-system = [ setuptools ];
  dependencies = [ biopython six ];
  doCheck = false;
}
