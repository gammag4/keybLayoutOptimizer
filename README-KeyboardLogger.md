## Logging keyboard and mouse events

Before you create a custom keyboard layout, you need to create a custom dataset from your own pc usage.

First, configure python environment:

```sh
python -m venv log_venv \
  && rm -rf ./keyboard && git clone https://github.com/boppreh/keyboard \
  && cd keyboard && git reset --hard d232de09bda50ecb5211ebcc59b85bc6da6aaa24 && cd .. \
  && rm -rf ./mouse && git clone https://github.com/boppreh/mouse \
  && cd mouse && git reset --hard 7b773393ed58824b1adf055963a2f9e379f52cc3 && cd .. \
  && source log_venv/bin/activate \
  && pip install ./keyboard \
  && pip install ./mouse
```

then run the script, where `log.json` will be the file where it will log keyboard and mouse events:

```sh
sudo su -c "source log_venv/bin/activate && python keyboard_mouse_logger.py log.json"
```
