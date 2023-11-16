#  Raw Data

In diesem Ordner befinden sich die Rohdaten, welche noch nicht verarbeitet wurden.
Unter keinen Umständen sollen hier Daten gespeichert werden, welche von DVC Stages erzeugt wurden.
Von DVC generierte files sollen im data/processed Ordner abgelegt werden.

Wenn neue Rohdaten hinzugefügt werden, dann müssen diese mittels `dvc add data/raw/{filename}` zur DVC Verwaltung hinzugefügt werden.
