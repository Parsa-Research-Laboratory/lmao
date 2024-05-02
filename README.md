## **L**ava **M**ulti-**A**gent **O**ptimization (LMAO)

Before installing LMAO, you must install the base [Lava](https://github.com/lava-nc/lava) library. If you would like to run the satellite scheduling problem, you must also install [Lava-Optimization](https://github.com/lava-nc/lava-optimization).

With `Lava` installed, LMAO can be installed as an editable Python package with the following command:

```python
python -m pip install .
```

There is a known issue with some versions of Scikit-Optimize where newer NumPy versions do not support `np.int` data structures. If you get this error, please `CTRL+F` within the library and replace all instances of `np.int` with `np.int32`.

---

### Sample Application

We have included the satellite scheduling problem from `Lava-Optimization` as a baseline application [here](./notebooks/demo_01_satellite_scheduler.ipynb). The notebook has all of the necessary infrastructure to highlight how you can wrap your problem in a function wrapper and connect it with LMAO.

If you run into any issues, please reach out to the maintainer.

---

### Authors

- Shay Snyder: [ssnyde9@gmu.edu](ssmnyde9@gmu.edu) (maintainer) (1)
- Derek Gobin (1)
- Victoria Clerico (1)
- Sumedh R. Risbud (2)
- Maryam Parsa (1)

(1) George Mason University (2) Intel Labs

### Acknowledgements

This work was supported by a generous gift the Intel corporation.

### Citation

If you find LMAO useful in your work, we would appreciate a citation :\)

```text
ADD ARXIV CITATION
```
