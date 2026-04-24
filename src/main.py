from model import Lesk

def main() -> None:
    lesk = Lesk()
    context = "Для расшифровки архивных документов криптографу потребовался дополнительный программный ключ, позволяющий корректно интерпретировать зашифрованные массивы данных."
    target  = "ключ"
    _, synonims, hyponims, hyperonims = lesk.disambiguate(context, target)
    print("Синонимы: ", synonims)
    print("Гипонимы: ", hyponims)
    print("Гиперонимы: ", hyperonims)
    
if __name__ == "__main__":
    main()