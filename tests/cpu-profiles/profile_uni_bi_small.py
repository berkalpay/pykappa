from pykappa.system import System

if __name__ == "__main__":
    system = System.from_kappa(
        {"A(l[.], r[.], u[.], d[.])": 200},
        [
            "A(l[.]), A(r[.]) <-> A(l[1]), A(r[1]) @ 25.0 {25.0}, 25.0",
            "A(u[.]), A(d[.]) <-> A(u[1]), A(d[1]) @ 25.0 {25.0}, 25.0",
        ],
    )
    while system.time < 1:
        system.update()
