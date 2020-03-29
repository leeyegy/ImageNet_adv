for epsilon in 0.00784 0.03137 0.06275
do 
python data_generator.py --attack_method JSMA --epsilon $epsilon
done 
