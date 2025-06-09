import marimo

__generated_with = "0.13.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell
def _():
    import zammad
    return (zammad,)


@app.cell
def _(zammad):
    dir(zammad)
    return


@app.cell
def _(zammad):
    zammad.fresh_tickets
    return


@app.cell
def _(zammad):
    zammad.fresh_tickets.__dict__
    return


@app.cell
def _(zammad):
    zammad.app
    return


@app.cell
def _(zammad):
    zammad.app.__dict__
    return


@app.cell
def _(zammad):
    dir(zammad.app)
    return


@app.cell
def _(zammad):
    zammad.app.run()
    return


@app.cell
def _(zammad):
    zammad.iterate_zammad_openai()
    return


if __name__ == "__main__":
    app.run()
