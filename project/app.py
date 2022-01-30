from flask import Flask, render_template, request, flash, redirect, url_for
from flask_wtf import FlaskForm, validators
from wtforms import IntegerField, SelectField, FileField
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from ml.generateModels import prepare_Data
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'

UPLOAD_FOLDER = 'ml/data/UPLOAD'
ALLOWED_EXT = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class GenModelForm(FlaskForm):
    file = FileField('File')
    batch_size1 = IntegerField('Batch Size')
    epochs1 = IntegerField('Epochs')
    output_dim = IntegerField('Output Dims')
    activation1 = SelectField('Activation Method', choices=[('relu', 'Relu'), ('elu', 'Elu'), ('softmax', 'Softmax')])
    activation2 = SelectField('Activation Method', choices=[('relu', 'Relu'), ('elu', 'Elu'), ('softmax', 'Softmax')])
    output_dims1 = IntegerField('Output Dims')
    batch_size2 = IntegerField('Batch Size')
    epochs2 = IntegerField('Epochs')
    activation3 = SelectField('Activation Method', choices=[('relu', 'Relu'), ('elu', 'Elu'), ('softmax', 'Softmax')])
    activation4 = SelectField('Activation Method', choices=[('relu', 'Relu'), ('elu', 'Elu'), ('softmax', 'Softmax')])
    activation5 = SelectField('Activation Method', choices=[('relu', 'Relu'), ('elu', 'Elu'), ('softmax', 'Softmax')])
    activation6 = SelectField('Activation Method', choices=[('relu', 'Relu'), ('elu', 'Elu'), ('softmax', 'Softmax')])
    output_dims2 = IntegerField('Output Dims')
    output_dims3 = IntegerField('Output Dims')
    output_dims4 = IntegerField('Output Dims')

# check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET', 'POST'])
def generate_model():
    form = GenModelForm()
    file = form.file.data
    batch_size1 = form.batch_size1.data
    epochs1 = form.epochs1.data
    activation1 = form.activation1.data
    activation2 = form.activation2.data
    output_dims1 = form.output_dims1.data
    batch_size2 = form.batch_size2.data
    epochs2 = form.epochs2.data
    activation3 = form.activation3.data
    activation4 = form.activation4.data
    activation5 = form.activation5.data
    activation6 = form.activation6.data
    output_dims2 = form.output_dims2.data
    output_dims3 = form.output_dims3.data
    output_dims4 = form.output_dims4.data

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # try to generate model 10 times
            model, iter = 'Error', 0
            while model == 'Error' and iter < 10:
                try:
                    model = prepare_Data(
                        app.config['UPLOAD_FOLDER']+f'/{filename}',
                        int(batch_size1),
                        int(epochs1),
                        activation1,
                        activation2,
                        int(output_dims1),
                        int(batch_size2),
                        int(epochs2),
                        activation3,
                        activation4,
                        activation5,
                        activation6,
                        int(output_dims2),
                        int(output_dims3),
                        int(output_dims4)
                    )
                except:
                   model = 'Error'
                   iter = iter + 1
            return render_template('results.html', model=model)
    return render_template('generate_model.html', form=form)

app.run(host='0.0.0.0', port=8098, debug=True)