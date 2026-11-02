from pathlib import Path

p = Path("tests/test_scientific_apis.py")
s = p.read_text().replace(
    "segmentation = Segmentation(id='cells', acquisition_id='a', labels=np.ones((11, 11), dtype=int))",
    "segmentation = Segmentation(id='cells', acquisition_id='a', labels=np.ones((11, 11), dtype=int),\n                                    metadata={'algorithm': 'cellpose', 'cellpose_version': '4.2', 'torch_version': '2.8'})",
)
p.write_text(s)
